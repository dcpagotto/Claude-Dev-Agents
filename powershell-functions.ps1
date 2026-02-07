# ⚡ Funções PowerShell - Claude Dev Agents
# Adicione estas funções ao seu perfil do PowerShell para atalhos rápidos

# Para adicionar ao perfil:
# 1. notepad $PROFILE
# 2. Cole o conteúdo deste arquivo
# 3. Salve e reinicie o PowerShell

# ==========================================
# Atalhos para Claude Code
# ==========================================

# Listar todos os agents
function cagents {
    claude /agents
}

# Ir para pasta de projetos
function projetos {
    Set-Location "C:\Users\dcpagotto\Documents\Projetos"
}

# Ir para pasta Claude Dev Agents
function cdagents {
    Set-Location "C:\Users\dcpagotto\Documents\Projetos\Claude-Dev-Agents"
}

# ==========================================
# Atalhos para Orchestrators
# ==========================================

# Tech Lead Orchestrator
function ctechlead {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-tech-lead-orchestrator and $task"
}

# Project Analyst
function canalyze {
    claude "use @agent-project-analyst and detect technology stack"
}

# Team Configurator
function cteam {
    claude "use @agent-team-configurator and optimize my project to best use the available subagents"
}

# ==========================================
# Atalhos para Core Team
# ==========================================

# Code Archaeologist
function cexplore {
    claude "use @agent-code-archaeologist and document this codebase"
}

# Code Reviewer
function creview {
    param([string]$file = ".")
    claude "use @agent-code-reviewer and review $file"
}

# Performance Optimizer
function coptimize {
    claude "use @agent-performance-optimizer and find bottlenecks"
}

# Documentation Specialist
function cdocs {
    param([Parameter(Mandatory=$true)][string]$type)
    claude "use @agent-documentation-specialist and create $type documentation"
}

# ==========================================
# Atalhos para Python Specialists
# ==========================================

# Python Expert
function cpython {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-python-expert and $task"
}

# FastAPI Expert
function cfastapi {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-fastapi-expert and $task"
}

# Django Expert
function cdjango {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-django-expert and $task"
}

# ML & Data Expert
function cml {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-ml-data-expert and $task"
}

# ==========================================
# Atalhos para Frontend
# ==========================================

# React Component Architect
function creact {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-react-component-architect and $task"
}

# Next.js Expert
function cnext {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-react-nextjs-expert and $task"
}

# Vue Component Architect
function cvue {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-vue-component-architect and $task"
}

# Tailwind CSS Expert
function ctailwind {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-tailwind-css-expert and $task"
}

# ==========================================
# Atalhos para Database & Deploy
# ==========================================

# Database Expert
function cdb {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-database-expert and $task"
}

# Docker Expert
function cdocker {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-docker-expert and $task"
}

# Kubernetes Expert
function ck8s {
    param([Parameter(Mandatory=$true)][string]$task)
    claude "use @agent-kubernetes-expert and $task"
}

# ==========================================
# Utilidades
# ==========================================

# Validar instalação
function ctest {
    & "C:\Users\dcpagotto\Documents\Projetos\Claude-Dev-Agents\teste-instalacao.ps1"
}

# Abrir documentação
function cdocs-open {
    param([string]$doc = "README-PT")
    $docPath = "C:\Users\dcpagotto\Documents\Projetos\Claude-Dev-Agents\$doc.md"
    if (Test-Path $docPath) {
        notepad $docPath
    } else {
        Write-Host "Documento não encontrado: $docPath" -ForegroundColor Red
        Write-Host "Documentos disponíveis:" -ForegroundColor Yellow
        Write-Host "  - README-PT" -ForegroundColor Cyan
        Write-Host "  - INSTALACAO-COMPLETA" -ForegroundColor Cyan
        Write-Host "  - ATALHOS-RAPIDOS" -ForegroundColor Cyan
    }
}

# Atualizar agents
function cagents-update {
    $currentDir = Get-Location
    Set-Location "C:\Users\dcpagotto\Documents\Projetos\Claude-Dev-Agents"
    Write-Host "Atualizando agents..." -ForegroundColor Yellow
    git pull origin main
    Write-Host "✅ Agents atualizados!" -ForegroundColor Green
    Set-Location $currentDir
}

# Mostrar ajuda
function chelp {
    Write-Host ""
    Write-Host "🚀 Claude Dev Agents - Comandos Rápidos" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📋 Comandos Gerais:" -ForegroundColor Yellow
    Write-Host "  cagents           - Listar todos os agents"
    Write-Host "  projetos          - Ir para pasta de projetos"
    Write-Host "  cdagents          - Ir para pasta Claude Dev Agents"
    Write-Host "  ctest             - Validar instalação"
    Write-Host "  cagents-update    - Atualizar agents do GitHub"
    Write-Host "  cdocs-open [doc]  - Abrir documentação"
    Write-Host "  chelp             - Mostrar esta ajuda"
    Write-Host ""
    Write-Host "🎭 Orchestrators:" -ForegroundColor Yellow
    Write-Host "  ctechlead '<task>'  - Tech Lead Orchestrator"
    Write-Host "  canalyze            - Analisar stack do projeto"
    Write-Host "  cteam               - Configurar time de AI"
    Write-Host ""
    Write-Host "🔧 Core Team:" -ForegroundColor Yellow
    Write-Host "  cexplore          - Documentar codebase"
    Write-Host "  creview [file]    - Revisar código"
    Write-Host "  coptimize         - Encontrar bottlenecks"
    Write-Host "  cdocs '<type>'    - Criar documentação"
    Write-Host ""
    Write-Host "🐍 Python:" -ForegroundColor Yellow
    Write-Host "  cpython '<task>'   - Python expert"
    Write-Host "  cfastapi '<task>'  - FastAPI expert"
    Write-Host "  cdjango '<task>'   - Django expert"
    Write-Host "  cml '<task>'       - ML & Data expert"
    Write-Host ""
    Write-Host "🎨 Frontend:" -ForegroundColor Yellow
    Write-Host "  creact '<task>'    - React expert"
    Write-Host "  cnext '<task>'     - Next.js expert"
    Write-Host "  cvue '<task>'      - Vue expert"
    Write-Host "  ctailwind '<task>' - Tailwind expert"
    Write-Host ""
    Write-Host "🗄️ Database & Deploy:" -ForegroundColor Yellow
    Write-Host "  cdb '<task>'       - Database expert"
    Write-Host "  cdocker '<task>'   - Docker expert"
    Write-Host "  ck8s '<task>'      - Kubernetes expert"
    Write-Host ""
    Write-Host "💡 Exemplos:" -ForegroundColor Yellow
    Write-Host "  ctechlead 'build authentication system'"
    Write-Host "  cfastapi 'create REST API for products'"
    Write-Host "  creact 'create dashboard component'"
    Write-Host "  cdocker 'containerize this application'"
    Write-Host ""
}

# ==========================================
# Mensagem de boas-vindas (opcional)
# ==========================================

Write-Host ""
Write-Host "✅ Claude Dev Agents carregado!" -ForegroundColor Green
Write-Host "   Digite 'chelp' para ver comandos disponíveis" -ForegroundColor Cyan
Write-Host ""
