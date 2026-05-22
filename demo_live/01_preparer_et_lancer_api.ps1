param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [int]$DashboardPort = 8502,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ApiPidFile = Join-Path $PSScriptRoot ".api_demo.pid"
$UiPidFile = Join-Path $PSScriptRoot ".streamlit_ui_demo.pid"
$DashboardPidFile = Join-Path $PSScriptRoot ".streamlit_dashboard_demo.pid"
$BaseUrl = "http://${HostAddress}:${ApiPort}"
$UiUrl = "http://${HostAddress}:${UiPort}"
$DashboardUrl = "http://${HostAddress}:${DashboardPort}"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Assert-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Commande '$Name' introuvable. $InstallHint"
    }
}

function Wait-ForDocker {
    Assert-Command "docker" "Installe Docker Desktop puis relance ce script."

    while ($true) {
        try {
            docker info *> $null
            Write-Host "Docker est disponible." -ForegroundColor Green
            return
        }
        catch {
            Write-Host "Docker Desktop ne semble pas demarre." -ForegroundColor Yellow
            Read-Host "Demarre Docker Desktop, attends qu'il soit pret, puis appuie sur Entree"
        }
    }
}

function Test-ApiHealth {
    try {
        $response = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 2
        return $response.status -eq "ok"
    }
    catch {
        return $false
    }
}

function Wait-ForApi {
    param([int]$TimeoutSeconds = 90)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-ApiHealth) {
            Write-Host "API disponible sur $BaseUrl" -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "L'API ne repond pas sur $BaseUrl apres $TimeoutSeconds secondes."
}

function Test-HttpEndpoint {
    param([string]$Url)

    try {
        Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 *> $null
        return $true
    }
    catch {
        return $false
    }
}

function Wait-ForHttpEndpoint {
    param(
        [string]$Url,
        [string]$Label,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpEndpoint $Url) {
            Write-Host "$Label disponible sur $Url" -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "$Label ne repond pas sur $Url apres $TimeoutSeconds secondes."
}

function Wait-ForPostgres {
    param([int]$TimeoutSeconds = 90)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $health = docker inspect -f "{{.State.Health.Status}}" projet8_monitoring_postgres 2>$null
        if ($health -eq "healthy") {
            Write-Host "PostgreSQL monitoring est disponible." -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "PostgreSQL monitoring n'est pas healthy apres $TimeoutSeconds secondes."
}

function Start-DemoProcess {
    param(
        [string]$Label,
        [string]$Command,
        [string]$PidFile
    )

    if (Test-Path $PidFile) {
        Remove-Item -LiteralPath $PidFile -Force
    }

    $process = Start-Process `
        -FilePath "powershell" `
        -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Command `
        -PassThru `
        -WindowStyle Hidden

    $process.Id | Out-File -FilePath $PidFile -Encoding ascii
    Write-Host "$Label lance en arriere-plan. PID : $($process.Id)"
}

Set-Location $RepoRoot

Write-Section "Preparation de la demo live Projet 8"
Write-Host "Depot : $RepoRoot"
Write-Host "URL API cible : $BaseUrl"
Write-Host "URL interface Streamlit API : $UiUrl"
Write-Host "URL dashboard monitoring : $DashboardUrl"

Write-Section "Verification des outils"
Assert-Command "uv" "Installe uv : https://docs.astral.sh/uv/getting-started/installation/"
Wait-ForDocker

Write-Section "Installation des dependances Python"
uv sync --extra dev

Write-Section "Verification de l'artefact modele"
uv run python scripts/check_model_load.py

Write-Section "Demarrage de PostgreSQL local pour le monitoring"
docker compose -f docker-compose.monitoring.yml up -d
Wait-ForPostgres

Write-Section "Demarrage de l'API FastAPI"
if (Test-ApiHealth) {
    Write-Host "Une API repond deja sur $BaseUrl. Aucun nouveau serveur n'est lance." -ForegroundColor Yellow
}
else {
    $apiCommand = "cd `"$RepoRoot`"; uv run uvicorn app.main:app --host $HostAddress --port $ApiPort"
    Start-DemoProcess -Label "Serveur API" -Command $apiCommand -PidFile $ApiPidFile
}

Wait-ForApi

Write-Section "Generation d'evenements de monitoring pour la demo"
$payload = Get-Content -Raw -Path "tests\payloads\valid_raw_client.json"
Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/predict/batch" `
    -Body $payload `
    -ContentType "application/json" `
    -TimeoutSec 60 *> $null
uv run python scripts/import_monitoring_logs_to_postgres.py --truncate
uv run python scripts/analyze_monitoring_logs.py --source postgres

Write-Section "Demarrage de l'interface Streamlit de scoring"
if (Test-HttpEndpoint $UiUrl) {
    Write-Host "L'interface Streamlit API repond deja sur $UiUrl." -ForegroundColor Yellow
}
else {
    $uiCommand = "cd `"$RepoRoot`"; `$env:API_BASE_URL=`"$BaseUrl`"; uv run streamlit run ui/streamlit_app.py --server.address $HostAddress --server.port $UiPort"
    Start-DemoProcess -Label "Interface Streamlit API" -Command $uiCommand -PidFile $UiPidFile
}
Wait-ForHttpEndpoint -Url $UiUrl -Label "Interface Streamlit API"

Write-Section "Demarrage du dashboard Streamlit de monitoring"
if (Test-HttpEndpoint $DashboardUrl) {
    Write-Host "Le dashboard monitoring repond deja sur $DashboardUrl." -ForegroundColor Yellow
}
else {
    $dashboardCommand = "cd `"$RepoRoot`"; uv run streamlit run dashboard/monitoring_dashboard.py --server.address $HostAddress --server.port $DashboardPort"
    Start-DemoProcess `
        -Label "Dashboard Streamlit monitoring" `
        -Command $dashboardCommand `
        -PidFile $DashboardPidFile
}
Wait-ForHttpEndpoint -Url $DashboardUrl -Label "Dashboard Streamlit monitoring"

Write-Section "Demo prete"
Write-Host "Swagger : $BaseUrl/docs"
Write-Host "Interface Streamlit API : $UiUrl"
Write-Host "Dashboard monitoring : $DashboardUrl"
Write-Host ""
Write-Host "Pour arreter PostgreSQL apres la demo :"
Write-Host "docker compose -f docker-compose.monitoring.yml down"
