param(
    [string]$BranchName = "",
    [string]$Remote = "origin",
    [switch]$StayOnCurrentBranch
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TriggerFile = Join-Path $PSScriptRoot "ci_cd_trigger.md"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Commande '$Name' introuvable."
    }
}

function Get-CurrentBranch {
    return (git branch --show-current).Trim()
}

function Test-CleanWorkingTree {
    $status = git status --porcelain
    return [string]::IsNullOrWhiteSpace($status)
}

function Test-CiPushBranch {
    param([string]$Name)

    return $Name -eq "main" -or $Name -eq "develop" -or $Name.StartsWith("feature/")
}

Set-Location $RepoRoot
Assert-Command "git"

Write-Section "Declenchement demo du pipeline CI/CD"
Write-Host "Depot : $RepoRoot"

if (-not (Test-CleanWorkingTree)) {
    git status --short
    throw "Working tree non propre. Commit, stash ou annule les changements avant de lancer ce script."
}

$initialBranch = Get-CurrentBranch
Write-Host "Branche courante : $initialBranch"

if ($StayOnCurrentBranch) {
    $targetBranch = $initialBranch
}
elseif ([string]::IsNullOrWhiteSpace($BranchName)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $targetBranch = "feature/demo-ci-cd-trigger-$timestamp"
}
else {
    $targetBranch = $BranchName
}

if (-not (Test-CiPushBranch $targetBranch)) {
    throw "La branche '$targetBranch' ne declenchera pas le workflow sur push. Utilise main, develop ou feature/**."
}

if ($targetBranch -ne $initialBranch) {
    Write-Section "Creation de la branche de demo"
    git switch -c $targetBranch
}

Write-Section "Modification minimale pour declencher la CI"
$now = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
$commitSha = (git rev-parse --short HEAD).Trim()

$content = @"
# Declencheur demo CI/CD

Ce fichier est modifie volontairement pour declencher le workflow GitHub Actions pendant une demo.

- Date locale : $now
- Branche : $targetBranch
- Commit de depart : $commitSha
"@

Set-Content -Path $TriggerFile -Value $content -Encoding utf8
git add $TriggerFile

if ([string]::IsNullOrWhiteSpace((git status --porcelain))) {
    Write-Host "Aucun changement detecte. Rien a commit." -ForegroundColor Yellow
    exit 0
}

Write-Section "Commit"
git commit -m "chore(ci): declencher le pipeline de demo"

Write-Section "Push"
git push -u $Remote $targetBranch

Write-Section "Pipeline lance"
$remoteUrl = (git remote get-url $Remote).Trim()
if ($remoteUrl -match "github\.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+)(\.git)?$") {
    $owner = $Matches.owner
    $repo = $Matches.repo
    Write-Host "Actions : https://github.com/$owner/$repo/actions"
    Write-Host "Branche : https://github.com/$owner/$repo/tree/$targetBranch"
}
else {
    Write-Host "Remote pousse : $remoteUrl"
}

Write-Host ""
Write-Host "Note : le deploiement Hugging Face ne part que sur un push main."
Write-Host "Ce script declenche surtout les jobs qualite/tests/build Docker via une branche feature/**."
