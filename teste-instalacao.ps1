# Script de Teste - Claude Dev Agents
# Execute em um NOVO terminal PowerShell

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "   Teste de Instalação - Claude Dev Agents" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Teste 1: Verificar se Claude Code está no PATH
Write-Host "[1/4] Verificando Claude Code..." -ForegroundColor Yellow
try {
    $claudeVersion = & claude --version 2>&1
    Write-Host "✅ Claude Code encontrado: $claudeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Claude Code não encontrado no PATH" -ForegroundColor Red
    Write-Host "   Abra um NOVO terminal para carregar o PATH atualizado" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Teste 2: Verificar diretório de agents
Write-Host "[2/4] Verificando diretório de agents..." -ForegroundColor Yellow
$agentsPath = "$env:USERPROFILE\.claude\agents\awesome-claude-agents"
if (Test-Path $agentsPath) {
    Write-Host "✅ Diretório de agents encontrado: $agentsPath" -ForegroundColor Green
} else {
    Write-Host "❌ Diretório de agents não encontrado" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Teste 3: Contar agents disponíveis
Write-Host "[3/4] Contando agents disponíveis..." -ForegroundColor Yellow
$agentCount = (Get-ChildItem -Path $agentsPath -Recurse -Filter "*.md").Count
Write-Host "✅ Total de agents encontrados: $agentCount" -ForegroundColor Green

Write-Host ""

# Teste 4: Listar categorias
Write-Host "[4/4] Listando categorias de agents..." -ForegroundColor Yellow
$categories = Get-ChildItem -Path $agentsPath -Directory | Select-Object -ExpandProperty Name
foreach ($category in $categories) {
    $count = (Get-ChildItem -Path "$agentsPath\$category" -Recurse -Filter "*.md").Count
    Write-Host "   📁 $category : $count agents" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "   ✅ Instalação validada com sucesso!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📝 Próximos passos:" -ForegroundColor Yellow
Write-Host "   1. Navegue até seu projeto:"
Write-Host "      cd C:\Users\dcpagotto\Documents\Projetos\seu-projeto"
Write-Host ""
Write-Host "   2. Liste todos os agents:"
Write-Host "      claude /agents"
Write-Host ""
Write-Host "   3. Configure seu projeto:"
Write-Host "      claude `"use @agent-team-configurator and optimize my project`""
Write-Host ""
Write-Host "   4. Comece a desenvolver:"
Write-Host "      claude `"use @agent-tech-lead-orchestrator and build a feature`""
Write-Host ""

Write-Host "📖 Documentação completa:" -ForegroundColor Cyan
Write-Host "   C:\Users\dcpagotto\Documents\Projetos\Claude-Dev-Agents\INSTALACAO-COMPLETA.md"
Write-Host ""
