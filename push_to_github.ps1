# Script to initialize git repository, create private GitHub repository, commit, and push
$ErrorActionPreference = "Stop"

$repoName = "insurance-underwriting-optimizer"
$token = "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"
$username = "satarabdus692-bot"
$projectDir = "c:\Users\Windows\Downloads\New folder\insurance-underwriting-optimizer"

Set-Location -Path $projectDir

Write-Host "1. Creating GitHub repository via API..."
$headers = @{
    "Authorization" = "token $token"
    "Accept"        = "application/vnd.github.v3+json"
    "User-Agent"    = "PowerShell-Script"
}
$body = @{
    name        = $repoName
    description = "Underwriting Loss Ratio Optimizer using French Motor TPL & CAS Schedule P data. Survival analysis, Chain Ladder & Bornhuetter-Ferguson reserving, Tweedie GLM, and segment profitability analysis."
    private     = $true
} | ConvertTo-Json

try {
    $repo = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json"
    Write-Host "Created repository: $($repo.html_url)"
} catch {
    Write-Host "Repository creation response / already exists: $_"
}

Write-Host "2. Initializing local git repository..."
if (-not (Test-Path ".git")) {
    git init
    git branch -M main
}

git config user.name "satarabdus692-bot"
git config user.email "satarabdus692-bot@users.noreply.github.com"

Write-Host "3. Adding all project files..."
git add .

Write-Host "4. Committing files..."
git commit -m "Initial commit: Insurance Underwriting Loss Ratio Optimizer complete actuarial suite"

$ErrorActionPreference = "Continue"

Write-Host "Setting remote origin and pushing..."
$remoteUrl = "https://$username`:$token@github.com/$username/$repoName.git"
git remote add origin $remoteUrl
git push -u origin main --force

Write-Host "Push complete! Verifying remote repository..."
Write-Host "Remote URL: https://github.com/$username/$repoName"
