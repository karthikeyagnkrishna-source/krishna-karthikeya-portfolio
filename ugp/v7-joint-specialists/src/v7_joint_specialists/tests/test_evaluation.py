import importlib.util
import unittest
from fixtures import fixture,torch

class ExportChecks(unittest.TestCase):
    def test_final_protocol_writes_completion_without_duplicate_keys(self):
        self.api()
        import evaluate_joint_specialists as e
        self.assertTrue(callable(getattr(e,'finalize_protocol',None)),'Final protocol construction must be exercised')
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'protocol.json'
            e.finalize_protocol(path,'signature',e.check_completion(list(range(96)),1,57600))
            record=json.loads(path.read_text())
            self.assertTrue(record['complete']);self.assertFalse(record['goal_achieved']);self.assertEqual(record['cases'],96)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluate_joint_specialists'),'Matched strict evaluation missing')
        from evaluate_joint_specialists import check_completion,export_variant
        return check_completion,export_variant

    def test_missing_or_duplicate_cases_cannot_complete(self):
        check,_=self.api()
        with self.assertRaises(ValueError):check(list(range(95)),0,57600)
        with self.assertRaises(ValueError):check([0]*96,0,57600)
        self.assertFalse(check(list(range(96)),1,57600)['goal_achieved'])
        self.assertTrue(check(list(range(96)),0,57600)['goal_achieved'])
        with self.assertRaises(ValueError):check(list(range(96)),0,57599)

    def test_exact_export_preserves_times_and_one_failure(self):
        _,export=self.api()
        from contracts import Targets,Predictions
        parent,inp,b,s=fixture(1)
        inp.base={k:v[:1] if k!='horizon' else v for k,v in inp.base.items()}
        t=Targets(b['target_leg'][:1].clone(),b['target_force'][:1].clone(),b['target_body'][:1],b['target_contact'][:1],torch.tensor([[10.1]],dtype=torch.float64),[('example',1)])
        p=Predictions(t.leg.clone(),t.force.clone(),t.body.clone(),t.contact.clone(),parent.fk(t.leg[...,:3])[0])
        t.leg[...,6]=0;p.leg[...,6]=.06
        meta=dict(split='familiar',environment='ForestEnv',trajectory='P2000',test_id=1,adaptation_gap_s=0,_target_r=p.r)
        rows=export(meta,'full',inp,t,p)
        self.assertEqual(len(rows['joint']),12);self.assertEqual(len(rows['strict']),60)
        haa=next(r for r in rows['joint'] if r['joint']=='HAA' and r['leg']=='FL')
        self.assertEqual(haa['target_time_s'],10.1);self.assertAlmostEqual(haa['anchor_time_s'],10)
        self.assertTrue(haa['torque_exact_zero']);self.assertFalse(haa['torque_strict_pass'])
        self.assertEqual(sum(not r['pass'] for r in rows['strict']),4)
        self.assertEqual(len(rows['force']),4)

if __name__=='__main__':unittest.main()
