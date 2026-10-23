param(
    [string]$SshHost = 'UGP_USER@SSH_HOST'
)

$ErrorActionPreference = 'Stop'
$storageRoot = '/path/to/ugp/V7F_4080_Run/storage'
$helper = Join-Path $PSScriptRoot 'stage_tartanground_run.sh'
$downloadRoot = Join-Path $PSScriptRoot 'relay_downloads'
$requiredNames = @('imu.zip','rosbags.zip','image_lcam_front.zip','depth_lcam_front.zip','metadata.zip')
$targets = @(
    @{ Environment = 'OldTownFall'; Run = 'P2005'; Kind = 'training' },
    @{ Environment = 'OldTownFall'; Run = 'P2006'; Kind = 'training' },
    @{ Environment = 'OldTownFall'; Run = 'P2007'; Kind = 'training' },
    @{ Environment = 'OldTownFall'; Run = 'P2001'; Kind = 'holdout' },
    @{ Environment = 'ForestEnv'; Run = 'P2001'; Kind = 'holdout' },
    @{ Environment = 'GreatMarsh'; Run = 'P2027'; Kind = 'holdout' },
    @{ Environment = 'SeasonalForestWinter'; Run = 'P2006'; Kind = 'holdout' }
)

New-Item -ItemType Directory -Force $downloadRoot | Out-Null

& scp -o BatchMode=no -o ConnectTimeout=20 $helper "${SshHost}:/tmp/codex_stage_tartanground_run.sh"
if ($LASTEXITCODE -ne 0) { throw 'Could not deploy the remote staging helper.' }

foreach ($target in $targets) {
    $environment = $target.Environment
    $run = $target.Run
    $kind = $target.Kind
    $archiveDir = "$storageRoot/relay_archives/$environment/$run"
    Write-Output "RUN_START $kind $environment/$run"

    & ssh -o BatchMode=no -o ConnectTimeout=20 $SshHost "mkdir -p '$archiveDir'"
    if ($LASTEXITCODE -ne 0) { throw "Could not create $archiveDir." }

    $apiUri = "https://huggingface.co/api/datasets/theairlabcmu/TartanGround/tree/main/${environment}/Data_anymal/${run}?recursive=true&expand=true"
    # Invoke-RestMethod already returns a top-level Object[] here. Wrapping the
    # call in @() would nest that array and turn each selected item into Object[].
    $repoItems = Invoke-RestMethod -Uri $apiUri -Method Get

    foreach ($name in $requiredNames) {
        $repoPath = "$environment/Data_anymal/$run/$name"
        $matches = @($repoItems | Where-Object { $_.path -eq $repoPath })
        if ($matches.Count -ne 1) {
            throw "Expected exactly one official metadata record for $repoPath; found $($matches.Count)."
        }

        $item = $matches[0]
        $expectedSize = [int64]$item.size
        $expectedSha256 = [string]$item.lfs.oid
        if ($expectedSha256.Length -ne 64) {
            throw "Official LFS SHA-256 is missing for $repoPath."
        }

        $remoteFile = "$archiveDir/$name"
        $remoteHash = ((& ssh -o BatchMode=no -o ConnectTimeout=20 $SshHost "if test -f '$remoteFile'; then sha256sum '$remoteFile' | cut -d ' ' -f1; fi") -join '').Trim()
        if ($LASTEXITCODE -ne 0) { throw "Could not inspect remote archive $remoteFile." }
        if ($remoteHash -eq $expectedSha256) {
            Write-Output "REMOTE_REUSE $repoPath"
            continue
        }
        if ($remoteHash) {
            $mismatch = "$remoteFile.mismatch.$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
            & ssh -o BatchMode=no -o ConnectTimeout=20 $SshHost "mv -- '$remoteFile' '$mismatch'"
            if ($LASTEXITCODE -ne 0) { throw "Could not preserve mismatched archive $remoteFile." }
            Write-Output "REMOTE_PRESERVED_MISMATCH $mismatch"
        }

        $localPart = Join-Path $downloadRoot "$environment-$run-$name.part"
        if ((Test-Path -LiteralPath $localPart) -and (Get-Item -LiteralPath $localPart).Length -gt $expectedSize) {
            $oversize = "$localPart.oversize.$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
            Move-Item -LiteralPath $localPart -Destination $oversize
            Write-Output "LOCAL_PRESERVED_OVERSIZE $oversize"
        }

        $url = "https://huggingface.co/datasets/theairlabcmu/TartanGround/resolve/main/${repoPath}?download=true"
        Write-Output "DOWNLOAD_START $repoPath bytes=$expectedSize"
        & curl.exe --fail --location --retry 6 --retry-all-errors --retry-delay 5 --continue-at - --output $localPart $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed for $repoPath; partial file was preserved." }

        $actualSize = (Get-Item -LiteralPath $localPart).Length
        if ($actualSize -ne $expectedSize) {
            throw "Size mismatch for ${repoPath}: expected $expectedSize, got $actualSize."
        }
        $localHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $localPart).Hash.ToLowerInvariant()
        if ($localHash -ne $expectedSha256) {
            $badHash = "$localPart.bad-sha256.$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
            Move-Item -LiteralPath $localPart -Destination $badHash
            throw "SHA-256 mismatch for $repoPath; bad file preserved as $badHash."
        }
        Write-Output "DOWNLOAD_VERIFIED $repoPath sha256=$localHash"

        $remotePart = "$remoteFile.part"
        & scp -o BatchMode=no -o ConnectTimeout=20 $localPart "${SshHost}:$remotePart"
        if ($LASTEXITCODE -ne 0) { throw "SSH transfer failed for $repoPath; local file was preserved." }

        $uploadedHash = ((& ssh -o BatchMode=no -o ConnectTimeout=20 $SshHost "sha256sum '$remotePart' | cut -d ' ' -f1") -join '').Trim()
        if ($LASTEXITCODE -ne 0 -or $uploadedHash -ne $expectedSha256) {
            throw "Remote SHA-256 mismatch for $repoPath; both copies were preserved."
        }
        & ssh -o BatchMode=no -o ConnectTimeout=20 $SshHost "mv -- '$remotePart' '$remoteFile'"
        if ($LASTEXITCODE -ne 0) { throw "Could not finalize remote archive $remoteFile." }

        Remove-Item -LiteralPath $localPart
        Write-Output "REMOTE_VERIFIED $repoPath"
    }

    & ssh -o BatchMode=no -o ConnectTimeout=20 $SshHost "V7_STORAGE_ROOT='$storageRoot' bash /tmp/codex_stage_tartanground_run.sh '$environment' '$run'"
    if ($LASTEXITCODE -ne 0) { throw "Remote extraction or validation failed for $environment/$run." }
    Write-Output "RUN_READY $kind $environment/$run"
}

Write-Output 'ALL_REQUIRED_RUNS_READY'
