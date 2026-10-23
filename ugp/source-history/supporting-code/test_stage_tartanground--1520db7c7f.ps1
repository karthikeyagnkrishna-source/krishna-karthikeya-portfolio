param(
    [string]$SshHost = 'UGP_USER@SSH_HOST'
)

$ErrorActionPreference = 'Stop'
$helper = Join-Path $PSScriptRoot 'stage_tartanground_run.sh'
if (-not (Test-Path -LiteralPath $helper)) {
    throw 'Missing stage_tartanground_run.sh implementation.'
}

$stamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$fixture = Join-Path $PSScriptRoot "stage_fixture_$stamp"
$content = Join-Path $fixture 'content'
$archives = Join-Path $fixture 'archives'
$remoteRoot = "/tmp/codex_stage_test_$stamp"
$remoteArchives = "$remoteRoot/relay_archives/TestEnv/PTEST"

New-Item -ItemType Directory -Force $content,$archives | Out-Null

$directories = @(
    @{ Name = 'imu'; File = 'sample.txt' },
    @{ Name = 'rosbags'; File = 'PTEST.bag' },
    @{ Name = 'image_lcam_front'; File = '000000_lcam_front.png' },
    @{ Name = 'depth_lcam_front'; File = '000000_lcam_front_depth.png' }
)
foreach ($entry in $directories) {
    $directory = Join-Path $content $entry.Name
    New-Item -ItemType Directory -Force $directory | Out-Null
    Set-Content -LiteralPath (Join-Path $directory $entry.File) -Value 'fixture'
}
Set-Content -LiteralPath (Join-Path $content 'pose_lcam_front.txt') -Value 'fixture'

Compress-Archive -LiteralPath (Join-Path $content 'imu') -DestinationPath (Join-Path $archives 'imu.zip')
Compress-Archive -LiteralPath (Join-Path $content 'rosbags') -DestinationPath (Join-Path $archives 'rosbags.zip')
Compress-Archive -LiteralPath (Join-Path $content 'image_lcam_front') -DestinationPath (Join-Path $archives 'image_lcam_front.zip')
Compress-Archive -LiteralPath (Join-Path $content 'depth_lcam_front') -DestinationPath (Join-Path $archives 'depth_lcam_front.zip')
Compress-Archive -LiteralPath (Join-Path $content 'pose_lcam_front.txt') -DestinationPath (Join-Path $archives 'metadata.zip')
Set-Content -NoNewline -LiteralPath (Join-Path $archives 'metadata-corrupt.zip') -Value 'not a zip'

& ssh -o BatchMode=no $SshHost "mkdir -p '$remoteArchives'"
if ($LASTEXITCODE -ne 0) { throw 'Could not create the remote test archive directory.' }

& scp -o BatchMode=no $helper "${SshHost}:/tmp/codex_stage_tartanground_run.sh"
if ($LASTEXITCODE -ne 0) { throw 'Could not copy the staging helper.' }

foreach ($name in 'imu.zip','rosbags.zip','image_lcam_front.zip','depth_lcam_front.zip') {
    & scp -o BatchMode=no (Join-Path $archives $name) "${SshHost}:$remoteArchives/$name"
    if ($LASTEXITCODE -ne 0) { throw "Could not copy test archive $name." }
}
& scp -o BatchMode=no (Join-Path $archives 'metadata-corrupt.zip') "${SshHost}:$remoteArchives/metadata.zip"
if ($LASTEXITCODE -ne 0) { throw 'Could not copy the corrupt test archive.' }

& ssh -o BatchMode=no $SshHost "V7_STORAGE_ROOT='$remoteRoot' bash /tmp/codex_stage_tartanground_run.sh TestEnv PTEST"
if ($LASTEXITCODE -eq 0) { throw 'Corrupt ZIP was accepted.' }

& ssh -o BatchMode=no $SshHost "test ! -e '$remoteRoot/raw_tartanground/TestEnv/Data_anymal/PTEST'"
if ($LASTEXITCODE -ne 0) { throw 'A corrupt archive exposed a partial target trajectory.' }

& scp -o BatchMode=no (Join-Path $archives 'metadata.zip') "${SshHost}:$remoteArchives/metadata.zip"
if ($LASTEXITCODE -ne 0) { throw 'Could not replace the corrupt test archive.' }

& ssh -o BatchMode=no $SshHost "V7_STORAGE_ROOT='$remoteRoot' bash /tmp/codex_stage_tartanground_run.sh TestEnv PTEST"
if ($LASTEXITCODE -ne 0) { throw 'Valid archives did not stage successfully.' }

$verification = "test -d '$remoteRoot/raw_tartanground/TestEnv/Data_anymal/PTEST/imu'" +
    " -a -f '$remoteRoot/raw_tartanground/TestEnv/Data_anymal/PTEST/rosbags/PTEST.bag'" +
    " -a -d '$remoteRoot/raw_tartanground/TestEnv/Data_anymal/PTEST/image_lcam_front'" +
    " -a -d '$remoteRoot/raw_tartanground/TestEnv/Data_anymal/PTEST/depth_lcam_front'" +
    " -a -f '$remoteRoot/raw_tartanground/TestEnv/Data_anymal/PTEST/pose_lcam_front.txt'" +
    " -a -f '$remoteArchives/metadata.zip'"
& ssh -o BatchMode=no $SshHost $verification
if ($LASTEXITCODE -ne 0) { throw 'Staged trajectory or preserved archive is incomplete.' }

'PASS: corrupt input remains hidden; valid input is staged atomically and archives are preserved.'
