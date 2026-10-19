# Run a VS Code task from Windows while MuJoCo and RCSSServerMJ stay inside WSL.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('audit', 'evaluate-walk', 'record-walk', 'evaluate-kick', 'record-kick', 'metrics', 'v034-smoke', 'v034-train-short', 'v034-evaluate', 'v034-status', 'v035-smoke', 'v035-match-10m', 'v035-match-viewer', 'v035-server-viewer', 'v035-status', 'v035-train-ippo', 'v036-train-short', 'v036-evaluate', 'v036-record', 'v036-status', 'v036-viewer', 'v037-continue-midfield', 'v037-evaluate-midfield', 'v037-record-midfield', 'v037-status', 'v037-viewer', 'v038-gait', 'v038-smoke', 'v038-teacher-baseline', 'v038-bootstrap-baseline', 'v038-train-stage0', 'v038-train-stage0-safe', 'v038-evaluate-stage0', 'v038-train-stage1-safe', 'v038-evaluate-stage1', 'v038-viewer', 'v038-record', 'v038-status')]
    [string]$Task
)

$ErrorActionPreference = 'Stop'

# Resolve paths from this script so the workspace may remain on the D drive.
$projectRoot = Split-Path -Parent $PSScriptRoot
$windowsRobotRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
$distroName = if ($env:ROBOT_SOCCER_WSL_DISTRO) { $env:ROBOT_SOCCER_WSL_DISTRO } else { 'Ubuntu-24.04' }
$wslRobotRoot = "\\wsl.localhost\$distroName\home\linh\robot_soccer"
$wslProjectRoot = Join-Path $wslRobotRoot 'training_runs\rcss_t1_training_20260706'
$linuxProjectRoot = '/home/linh/robot_soccer/training_runs/rcss_t1_training_20260706'

function Wait-WslProjectShare {
    & wsl.exe -d $distroName -- bash -lc "test -d '$linuxProjectRoot'"
    if ($LASTEXITCODE -ne 0) {
        throw "Linux project is missing in distro '$distroName': $linuxProjectRoot"
    }

    foreach ($attempt in 1..20) {
        if (Test-Path -LiteralPath $wslProjectRoot) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "WSL project share did not become ready: $wslProjectRoot"
}

# A stopped distro may not expose \\wsl.localhost immediately.  Wake it through
# wsl.exe first, then wait for the file share before synchronizing Windows files.
Wait-WslProjectShare

# Sync only the active v032/v033 source layers. Historical runs stay in archive.
foreach ($folder in @('locomotion', 'configs', 'scripts', 'notes', 'learning', 'records', 'experiments', '.vscode')) {
    $source = Join-Path $projectRoot $folder
    $target = Join-Path $wslProjectRoot $folder
    if (Test-Path -LiteralPath $source) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force
    }
}

# Keep the three learning documents identical in Windows and WSL.
foreach ($document in @('README.md', 'PROJECT_GUIDE.md', 'CODE_READING_GUIDE.md')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $document) -Destination (Join-Path $wslProjectRoot $document) -Force
}

# Copy the official 13 MB Booster policy once when WSL does not have it yet.
$officialSource = Join-Path $windowsRobotRoot 'third_party\booster_gym'
$officialTarget = Join-Path $wslRobotRoot 'third_party\booster_gym'
if (-not (Test-Path -LiteralPath (Join-Path $officialTarget 'deploy\models\T1.pt'))) {
    if (-not (Test-Path -LiteralPath $officialSource)) {
        throw "Official Booster source is missing: $officialSource"
    }
    New-Item -ItemType Directory -Path $officialTarget -Force | Out-Null
    Copy-Item -Path (Join-Path $officialSource '*') -Destination $officialTarget -Recurse -Force
}

# Execute with the existing WSL virtual environment.
& wsl.exe -d $distroName -- bash -lc "cd $linuxProjectRoot && bash scripts/vscode_entrypoint.sh $Task"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

# Copy generated metrics and videos back to the D-drive workspace for VS Code.
Wait-WslProjectShare
foreach ($folder in @('records', 'videos')) {
    $source = Join-Path $wslProjectRoot $folder
    $target = Join-Path $projectRoot $folder
    if (Test-Path -LiteralPath $source) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force
    }
}

# v034 uses non-overwriting numbered artifacts. Copy only files that are not
# already present on Windows so a later task cannot replace historical evidence.
$v034Source = Join-Path $wslProjectRoot 'experiments\v034_dynamic_single_leg_kick'
$v034Target = Join-Path $projectRoot 'experiments\v034_dynamic_single_leg_kick'
foreach ($folder in @('checkpoints', 'logs', 'metrics', 'videos')) {
    $source = Join-Path $v034Source $folder
    $target = Join-Path $v034Target $folder
    if (Test-Path -LiteralPath $source) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Get-ChildItem -LiteralPath $source -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($source.Length).TrimStart('\')
            $destination = Join-Path $target $relative
            if (-not (Test-Path -LiteralPath $destination)) {
                $destinationDirectory = [System.IO.Path]::GetDirectoryName($destination)
                if ($destinationDirectory) {
                    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
                }
                Copy-Item -LiteralPath $_.FullName -Destination $destination
            }
        }
    }
}

# Copy only new v035 artifacts back to Windows; never replace historical evidence.
foreach ($artifactFolder in @('checkpoints', 'metrics', 'matches')) {
    $artifactSource = Join-Path (Join-Path $wslProjectRoot 'experiments\v035_7v7_self_play') $artifactFolder
    $artifactTarget = Join-Path (Join-Path $projectRoot 'experiments\v035_7v7_self_play') $artifactFolder
    if (Test-Path -LiteralPath $artifactSource) {
        New-Item -ItemType Directory -Path $artifactTarget -Force | Out-Null
        Get-ChildItem -LiteralPath $artifactSource -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($artifactSource.Length).TrimStart('\')
            $destination = Join-Path $artifactTarget $relative
            if (-not (Test-Path -LiteralPath $destination)) {
                $destinationDirectory = [System.IO.Path]::GetDirectoryName($destination)
                if ($destinationDirectory) {
                    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
                }
                Copy-Item -LiteralPath $_.FullName -Destination $destination
            }
        }
    }
}

# v036 keeps the trained single-player scoring model, fixed-seed metrics and video together.
foreach ($artifactFolder in @('checkpoints', 'metrics', 'videos')) {
    $artifactSource = Join-Path (Join-Path $wslProjectRoot 'experiments\v036_single_robot_score') $artifactFolder
    $artifactTarget = Join-Path (Join-Path $projectRoot 'experiments\v036_single_robot_score') $artifactFolder
    if (Test-Path -LiteralPath $artifactSource) {
        New-Item -ItemType Directory -Path $artifactTarget -Force | Out-Null
        Get-ChildItem -LiteralPath $artifactSource -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($artifactSource.Length).TrimStart('\')
            $destination = Join-Path $artifactTarget $relative
            if (-not (Test-Path -LiteralPath $destination)) {
                $destinationDirectory = [System.IO.Path]::GetDirectoryName($destination)
                if ($destinationDirectory) {
                    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
                }
                Copy-Item -LiteralPath $_.FullName -Destination $destination
            }
        }
    }
}

# v038 uses separate validation/test evidence and never overwrites a numbered artifact.
foreach ($artifactFolder in @('checkpoints', 'metrics', 'videos')) {
    $artifactSource = Join-Path (Join-Path $wslProjectRoot 'experiments\v038_high_speed_dribble') $artifactFolder
    $artifactTarget = Join-Path (Join-Path $projectRoot 'experiments\v038_high_speed_dribble') $artifactFolder
    if (Test-Path -LiteralPath $artifactSource) {
        New-Item -ItemType Directory -Path $artifactTarget -Force | Out-Null
        Get-ChildItem -LiteralPath $artifactSource -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($artifactSource.Length).TrimStart('\')
            $destination = Join-Path $artifactTarget $relative
            if (-not (Test-Path -LiteralPath $destination)) {
                $destinationDirectory = [System.IO.Path]::GetDirectoryName($destination)
                if ($destinationDirectory) {
                    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
                }
                Copy-Item -LiteralPath $_.FullName -Destination $destination
            }
        }
    }
}

# v037 keeps every numbered curriculum checkpoint and its fixed-seed evidence.
foreach ($artifactFolder in @('checkpoints', 'metrics', 'videos')) {
    $artifactSource = Join-Path (Join-Path $wslProjectRoot 'experiments\v037_midfield_dribble_score') $artifactFolder
    $artifactTarget = Join-Path (Join-Path $projectRoot 'experiments\v037_midfield_dribble_score') $artifactFolder
    if (Test-Path -LiteralPath $artifactSource) {
        New-Item -ItemType Directory -Path $artifactTarget -Force | Out-Null
        Get-ChildItem -LiteralPath $artifactSource -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($artifactSource.Length).TrimStart('\')
            $destination = Join-Path $artifactTarget $relative
            if (-not (Test-Path -LiteralPath $destination)) {
                $destinationDirectory = [System.IO.Path]::GetDirectoryName($destination)
                if ($destinationDirectory) {
                    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
                }
                Copy-Item -LiteralPath $_.FullName -Destination $destination
            }
        }
    }
}
