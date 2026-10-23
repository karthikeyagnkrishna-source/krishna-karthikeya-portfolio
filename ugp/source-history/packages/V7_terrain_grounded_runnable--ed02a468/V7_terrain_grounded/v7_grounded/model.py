"""V7 grounded world model. No future RGB-D, true footprints or future forces enter forward()."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import world_to_robot,project_future_patches

@dataclass
class V7Config:
    proprio_dim: int = 48
    feature_dim: int = 0  # optional registered DINO/ConvNeXt/PTv3/WAFT concat dimension
    action_dim: int = 16
    hidden: int = 320
    heads: int = 10
    spatial_layers: int = 8
    temporal_layers: int = 2
    mlp_ratio: int = 7
    horizons: int = 10
    legs: int = 4
    patch_reach_m: float = 1.0
    contact_sigma_m: float = .35
    step_seconds: float = .1
    world_delta_limit_m: float = .035


def mlp(inp:int,out:int,hidden:int) -> nn.Module:
    return nn.Sequential(nn.Linear(inp,hidden),nn.GELU(),nn.Linear(hidden,out))

class V7GroundedWorldModel(nn.Module):
    """Five-frame persistent patch memory -> action-conditioned terrain evolution ->
    leg conditioned cross attention -> contact response -> terrain-conditioned body.

    Evolving world coordinates is physical terrain deformation; world_to_robot on the
    same patch locations handles apparent image motion due to base egomotion.
    All heads are trained jointly. Their predictions, not labels, connect horizons.
    """
    def __init__(self,cfg:V7Config):
        super().__init__()
        if cfg.hidden%cfg.heads: raise ValueError('hidden must divide heads')
        self.cfg=cfg; d=cfg.hidden
        self.obs_embed=mlp(4+3+1+1,d,d)
        self.extra_vision=mlp(cfg.feature_dim,d,d) if cfg.feature_dim>0 else None
        self.extra_history=mlp(cfg.feature_dim,d,d) if cfg.feature_dim>0 else None
        self.time_embed=nn.Embedding(5,d)
        self.patch_pos=mlp(3,d,d)
        self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,cfg.heads,d*cfg.mlp_ratio,0.0,batch_first=True,norm_first=True),cfg.temporal_layers,enable_nested_tensor=False)
        self.spatial=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,cfg.heads,d*cfg.mlp_ratio,0.0,batch_first=True,norm_first=True),cfg.spatial_layers,enable_nested_tensor=False)
        self.future_spatial=nn.TransformerEncoderLayer(d,cfg.heads,d*cfg.mlp_ratio,0.0,batch_first=True,norm_first=True)
        self.proprio_emb=mlp(cfg.proprio_dim,d,d)
        self.action_emb=mlp(cfg.action_dim,d,d)
        self.action_history=nn.GRU(d,d,batch_first=True)
        self.ego_head=mlp(3*d,6,d)
        self.ego_scale=nn.Parameter(torch.tensor([.20,.20,.14,.10,.10,.17]))
        # Each TOUCHDOWN event keeps its own actual RGB, geometry, patch ID and physical response.
        self.hist_embed=mlp(3+3+4+6+1+4+3+cfg.proprio_dim,d,d)
        self.hist_time=mlp(1,d,d)
        self.hist_match=nn.Linear(d,d)
        self.leg_index=nn.Embedding(cfg.legs,d)
        self.leg_query=mlp(3+3+1+1,d,d)
        # Four explicit leg-specific action FiLM modulators (no previous-action input).
        self.leg_action_film=nn.ModuleList([mlp(d,2*d,d) for _ in range(cfg.legs)])
        self.memory_attn=nn.MultiheadAttention(d,cfg.heads,batch_first=True)
        self.future_patch_embed=mlp(3+4+3,d,d)
        self.geom_direct=mlp(2*d+1,7,d) # world xyz and RGB deltas + log sigma
        self.geom_step=mlp(2*d+1,6,d) # contact-conditioned incremental xyz + RGB
        self.leg_terrain_query=mlp(2*d,d,d)
        self.foot_offset=mlp(3*d+3,3,d)
        self.contact_head=mlp(3*d+3,1,d)
        self.force_head=mlp(3*d+3,3,d)
        self.body_head=mlp(2*d+cfg.legs*(3+3+1),cfg.proprio_dim,d)
        self.log_force_sigma=mlp(2*d,3,d)

    def _encode_patches(self,x:dict[str,torch.Tensor]):
        rgbd=x['rgbd'];xyz=x['xyz_world'];seen=x['valid'].bool()
        B,T,P,_=rgbd.shape;d=self.cfg.hidden
        if T!=5: raise ValueError('five causal visual frames required')
        if xyz.shape!=(B,T,P,3) or seen.shape!=(B,T,P):raise ValueError('bad geometry/valid shapes')
        times=torch.arange(T,device=xyz.device).view(1,T,1,1).expand(B,-1,P,-1)
        # Patch positional identity is WORLD registered, not an image pixel index.
        z=self.obs_embed(torch.cat([rgbd,xyz,times.to(xyz.dtype)/4,seen[...,None].to(xyz.dtype)],-1))
        if self.extra_vision is not None:
            if 'patch_features' not in x: raise KeyError('configured external visual features missing; no silent fallback')
            if x['patch_features'].shape!=(B,T,P,self.cfg.feature_dim): raise ValueError('bad patch_features shape')
            z=z+self.extra_vision(x['patch_features'])
        elif 'patch_features' in x: raise ValueError('set config.feature_dim for supplied patch_features')
        z=z+self.time_embed(torch.arange(T,device=z.device))[None,:,None,:]
        z=z.permute(0,2,1,3).reshape(B*P,T,d)
        missing=~seen.permute(0,2,1).reshape(B*P,T)
        all_missing=missing.all(-1)
        missing=missing.clone();missing[all_missing,0]=False
        z=self.temporal(z,src_key_padding_mask=missing)
        last=(~missing).long().sum(-1)-1  # choose final chronological valid slot via explicit indexing below
        chronological=(~missing).long()*torch.arange(1,T+1,device=z.device)[None,:]
        last=chronological.argmax(-1)
        z=z[torch.arange(B*P,device=z.device),last].reshape(B,P,d)
        visible=seen.any(1)
        if not visible.any(1).all():
            raise ValueError('No measured terrain patches in one or more samples; refusing to hallucinate a complete map')
        z=z*visible[...,None]
        ids=x['patch_ids']
        if ids.shape!=(B,T,P) or not torch.equal(ids,ids[:,-1:,:].expand_as(ids)):
            raise ValueError('patch IDs must match across all five registered frames')
        # Use most recent observed XYZ/RGB, not zero-filled future images.
        valid_idx=seen.long()*torch.arange(1,T+1,device=z.device)[None,:,None]
        last_idx=valid_idx.argmax(1)
        ix=last_idx[:,None,:,None]
        last_xyz=xyz.gather(1,ix.expand(-1,-1,-1,3)).squeeze(1)
        last_rgb=rgbd.gather(1,ix.expand(-1,-1,-1,4)).squeeze(1)
        z=z+self.patch_pos(last_xyz)
        z=z*visible[...,None]
        safe_mask=(~visible).clone();all_bad=safe_mask.all(-1)
        safe_mask[all_bad,0]=False
        z=self.spatial(z,src_key_padding_mask=safe_mask)*visible[...,None]
        return z,last_xyz,last_rgb,visible,ids[:,-1]

    def _history_memory(self,x:dict[str,torch.Tensor],patch_z:torch.Tensor,patch_ids:torch.Tensor):
        B,L,K,_=x['history_xyz_world'].shape;d=self.cfg.hidden
        if L!=self.cfg.legs:raise ValueError('expected four legs')
        hfeat=torch.cat([x['history_xyz_world'],x['history_rgb'],x['history_geometry'],x['history_response'],x['history_age'][...,None],
                         x['history_delta_rgbd'],x['history_delta_xyz'],x['history_delta_proprio']],-1)
        h=self.hist_embed(hfeat)+self.hist_time(x['history_age'][...,None])
        if self.extra_history is not None:
            if 'history_token_delta' not in x: raise KeyError('registered past visual feature changes missing')
            h=h+self.extra_history(x['history_token_delta'])
        h=h+self.leg_index(torch.arange(L,device=h.device))[None,:,None,:]
        # Explicit lookup from each touchdown's exact WORLD patch ID, if still in current map.
        exact=(x['history_patch_ids'][...,None]==patch_ids[:,None,None,:]) & x['history_valid'][...,None].bool()
        linked=torch.einsum('blkp,bpd->blkd',exact.to(h.dtype),patch_z)
        h=h+self.hist_match(linked)*exact.any(-1)[...,None]
        return h

    def forward(self,x:dict[str,torch.Tensor]) -> dict[str,torch.Tensor]:
        cfg=self.cfg
        z,world_xyz,rgb,patch_valid,ids=self._encode_patches(x)
        B,P,d=z.shape;L=cfg.legs;H=cfg.horizons
        actions=x['actions'];assert actions.shape==(B,H,cfg.action_dim), 'actions must be candidate controls known at t'
        prop=x['proprio'];assert prop.shape==(B,5,cfg.proprio_dim)
        current_pose=x['base_pose'];assert current_pose.shape==(B,6)
        query=self.leg_query(torch.cat([x['foot_xyz_world'],x['foot_velocity_world'],x['foot_contact'][...,None],x['gait_phase'][...,None]],-1))
        query=query+self.leg_index(torch.arange(L,device=z.device))[None,:,:]
        history=self._history_memory(x,z,ids)
        history=history.reshape(B*L,-1,d)
        hist_mask=~x['history_valid'].reshape(B*L,-1).bool()
        hist_all=hist_mask.all(-1); hist_mask=hist_mask.clone();hist_mask[hist_all,0]=False
        history=history.clone();history[hist_all]=0
        memory,_=self.memory_attn(query.reshape(B*L,1,d),history,history,key_padding_mask=hist_mask,need_weights=False)
        memory=memory.reshape(B,L,d)*~hist_all.reshape(B,L,1)
        terrain_global=(z*patch_valid[...,None]).sum(1)/patch_valid.sum(1,keepdim=True).clamp(min=1)
        pctx=self.proprio_emb(prop[:,-1]);a=self.action_emb(actions)
        action_context,_=self.action_history(a)
        # The action can affect the body ONLY by changing egomotion, terrain and contacts.
        future=[];direct=[];f_rgb=[];d_rgb=[];foot=[];patchlog=[];forces=[];contacts=[];bodies=[];poses=[];patchout=[];sigmas=[];f_sigmas=[];uvd=[]
        prev_xyz=world_xyz;prev_rgb=rgb;prev_contact_xyz=x['foot_xyz_world'];prev_contact_p=x['foot_contact'];
        prev_contact_force=torch.ones_like(prev_contact_p)
        pose=current_pose;prev_body=prop[:,-1]
        camera_ok='intrinsics' in x and 'camera_from_base' in x
        for step in range(H):
            ac=action_context[:,step]
            ctx=terrain_global+ac
            # Ego motion: rigid robot transform; DOES NOT physically move the world.
            pose_delta=self.ego_head(torch.cat([pctx,ctx,ac],-1))*self.ego_scale.to(z.device)
            pose=pose+pose_delta
            rel_to_contacts=prev_xyz[:,:,None,:]-prev_contact_xyz[:,None,:,:]
            contact_kernel=torch.exp(-rel_to_contacts.square().sum(-1)/(2*cfg.contact_sigma_m**2))
            contact_map=(contact_kernel*(prev_contact_p*prev_contact_force)[:,None,:]).sum(-1).clamp(0,1)
            # Trainable contact-induced changes; mask disallows arbitrary map-wide deformation.
            geo_in=torch.cat([z,ctx[:,None,:].expand(-1,P,-1),contact_map[...,None]],-1)
            step_delta=torch.tanh(self.geom_step(geo_in))*contact_map[...,None]
            step_delta=step_delta*cfg.world_delta_limit_m
            xyz_roll=prev_xyz+step_delta[...,:3]
            rgb_roll=(prev_rgb[:,:,:3]+step_delta[...,3:6]).clamp(0,1)
            direct_delta=self.geom_direct(geo_in)
            direct_xyz=world_xyz+torch.tanh(direct_delta[...,:3])*cfg.world_delta_limit_m*(step+1)*contact_map[...,None]
            direct_color=(rgb[:,:,:3]+torch.tanh(direct_delta[...,3:6])*.08*(step+1)*contact_map[...,None]).clamp(0,1)
            # Direct anchored multi-horizon estimate + physically propagated contact rollout.
            evolved_xyz=.5*xyz_roll+.5*direct_xyz
            evolved_rgb=.5*rgb_roll+.5*direct_color
            relative_xyz=world_to_robot(evolved_xyz,pose)
            future_depth=(project_future_patches(evolved_xyz,pose,x['camera_from_base'],x['intrinsics'])[...,2:3]
                          if camera_ok else relative_xyz.norm(dim=-1,keepdim=True))
            future_token=z+self.future_patch_embed(torch.cat([relative_xyz,evolved_rgb,future_depth,evolved_xyz-world_xyz],-1))
            pmask=(~patch_valid).clone();pmask[pmask.all(-1),0]=False
            future_token=self.future_spatial(future_token,src_key_padding_mask=pmask)
            future_token=future_token*patch_valid[...,None]
            film=torch.stack([self.leg_action_film[leg](ac) for leg in range(L)],dim=1)
            gamma,beta=film.chunk(2,-1)
            film_query=query*(1+.1*torch.tanh(gamma))+.1*beta
            legq=self.leg_terrain_query(torch.cat([film_query+ac[:,None,:],memory],-1))
            # One leg-specific cross-attention over reconstructed FUTURE patches.
            logits=torch.einsum('bld,bpd->blp',legq,future_token)*d**-.5
            planned=x['foot_xyz_world']+x['foot_velocity_world']*(step+1)*cfg.step_seconds+pose_delta[:,None,:3]
            dist=torch.cdist(planned,evolved_xyz)
            allowed=(dist<cfg.patch_reach_m)&patch_valid[:,None,:]
            nearest=(dist.masked_fill(~patch_valid[:,None,:],1e6)).argmin(-1)
            none=~allowed.any(-1)
            allowed=allowed.scatter(-1,nearest[...,None],True)
            logits=logits.masked_fill(~allowed,-1e4)
            weights=F.softmax(logits,-1)
            terrain_ctx=torch.einsum('blp,bpd->bld',weights,future_token)
            xyz_ctx=torch.einsum('blp,bpc->blc',weights,evolved_xyz)
            # Multiplicative terrain gating: memory and leg state cannot bypass FUTURE terrain.
            interaction=terrain_ctx*torch.sigmoid(legq+memory)
            mix=torch.cat([interaction,terrain_ctx,memory,xyz_ctx],-1)
            xyz_touch=xyz_ctx+.10*torch.tanh(self.foot_offset(mix))
            contact_logit=self.contact_head(mix).squeeze(-1)
            force=self.force_head(mix)
            f_sigma=self.log_force_sigma(torch.cat([interaction,terrain_ctx],-1)).clamp(-8,4)
            prob=torch.sigmoid(contact_logit)
            future_global=(future_token*patch_valid[...,None]).sum(1)/patch_valid.sum(1,keepdim=True).clamp(min=1)
            base_input=torch.cat([future_global,interaction.mean(1),torch.cat([xyz_touch,force,prob[...,None]],-1).reshape(B,-1)],-1)
            # Only evolved-terrain -> foothold/contact features reach the state head.
            body_delta=self.body_head(base_input)
            body=prop[:,-1]+body_delta
            future.append(evolved_xyz);direct.append(direct_xyz)
            f_rgb.append(evolved_rgb);d_rgb.append(direct_color)
            foot.append(xyz_touch);patchlog.append(logits);forces.append(force)
            contacts.append(contact_logit);bodies.append(body);poses.append(pose)
            patchout.append(ids.gather(-1,logits.argmax(-1)));sigmas.append(direct_delta[...,6].clamp(-7,3));f_sigmas.append(f_sigma)
            if camera_ok: uvd.append(project_future_patches(evolved_xyz,pose,x['camera_from_base'],x['intrinsics']))
            prev_xyz=evolved_xyz;prev_rgb=torch.cat([evolved_rgb,rgb[...,3:4]],-1)
            # Next predictions use predicted contacts, NOT measured future targets.
            prev_contact_xyz=xyz_touch;prev_contact_p=prob;prev_contact_force=force.norm(dim=-1).sigmoid()
        out=dict(future_xyz_world=torch.stack(future,1),direct_xyz_world=torch.stack(direct,1),
                 future_rgb=torch.stack(f_rgb,1),direct_rgb=torch.stack(d_rgb,1),
                 foot_xyz_world=torch.stack(foot,1),patch_logits=torch.stack(patchlog,1),
                 predicted_patch_ids=torch.stack(patchout,1),force=torch.stack(forces,1),
                 contact_logits=torch.stack(contacts,1),body=torch.stack(bodies,1),
                 base_pose=torch.stack(poses,1),patch_ids=ids,patch_valid=patch_valid,
                 geometry_log_sigma=torch.stack(sigmas,1),force_log_sigma=torch.stack(f_sigmas,1))
        if camera_ok:out['projected_uvd']=torch.stack(uvd,1)
        return out
