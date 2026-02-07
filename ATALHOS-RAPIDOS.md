# 🚀 Atalhos Rápidos - Claude Dev Agents

## 📋 Comandos Essenciais

### Listar Todos os Agents
```bash
claude /agents
```

### Ver Ajuda do Claude Code
```bash
claude --help
```

---

## 🎯 Workflows Rápidos

### 🆕 Inicializar Novo Projeto
```bash
# 1. Criar e entrar no diretório
cd C:\Users\dcpagotto\Documents\Projetos\
mkdir meu-novo-projeto
cd meu-novo-projeto

# 2. Configurar o time de AI
claude "use @agent-team-configurator and optimize my project to best use the available subagents."
```

### 🔍 Analisar Projeto Existente
```bash
# Detectar stack tecnológico
claude "use @agent-project-analyst and detect my technology stack"

# Documentar código existente
claude "use @agent-code-archaeologist and document this codebase"
```

### 🏗️ Desenvolvimento com Orchestrator
```bash
# Feature completa com coordenação automática
claude "use @agent-tech-lead-orchestrator and build [descrição da feature]"

# Exemplos:
claude "use @agent-tech-lead-orchestrator and build a user authentication system"
claude "use @agent-tech-lead-orchestrator and create a REST API for product management"
claude "use @agent-tech-lead-orchestrator and implement payment integration"
```

---

## 🔧 Agents por Tarefa

### 🐍 Python
```bash
# Expert geral Python
claude "use @agent-python-expert and [tarefa]"

# FastAPI
claude "use @agent-fastapi-expert and create a CRUD API"

# Django
claude "use @agent-django-expert and build an admin panel"

# Machine Learning
claude "use @agent-ml-data-expert and analyze this dataset"

# Web Scraping
claude "use @agent-web-scraping-expert and scrape [website]"

# Performance
claude "use @agent-performance-expert and optimize this code"

# Testing
claude "use @agent-testing-expert and create unit tests"

# Security
claude "use @agent-security-expert and audit security vulnerabilities"

# DevOps
claude "use @agent-devops-cicd-expert and setup CI/CD pipeline"
```

### 🗄️ Banco de Dados
```bash
# Otimização de queries
claude "use @agent-database-expert and optimize these SQL queries"

# Design de schema
claude "use @agent-database-expert and design database schema for [feature]"

# Migrations
claude "use @agent-database-expert and create migration for [changes]"
```

### 🚀 Deploy
```bash
# Docker
claude "use @agent-docker-expert and containerize this application"
claude "use @agent-docker-expert and create docker-compose for development"

# Kubernetes
claude "use @agent-kubernetes-expert and create k8s deployment"
claude "use @agent-kubernetes-expert and setup helm chart"
```

### 🎨 Frontend

#### React
```bash
# Componentes
claude "use @agent-react-component-architect and create [component]"

# Next.js
claude "use @agent-react-nextjs-expert and setup SSR page"
```

#### Vue
```bash
# Vue 3
claude "use @agent-vue-component-architect and create [component]"

# Nuxt
claude "use @agent-vue-nuxt-expert and setup Nuxt project"

# State Management
claude "use @agent-vue-state-manager and implement Pinia store"
```

#### Styling
```bash
# Tailwind
claude "use @agent-tailwind-css-expert and style this component"
```

### 🔨 Backend

#### Laravel
```bash
claude "use @agent-laravel-backend-expert and create [feature]"
claude "use @agent-laravel-eloquent-expert and optimize queries"
```

#### Django
```bash
claude "use @agent-django-backend-expert and create [feature]"
claude "use @agent-django-api-developer and create REST API"
claude "use @agent-django-orm-expert and optimize database queries"
```

#### Rails
```bash
claude "use @agent-rails-backend-expert and create [feature]"
claude "use @agent-rails-api-developer and create API endpoints"
claude "use @agent-rails-activerecord-expert and optimize queries"
```

### 📱 Mobile
```bash
# Android/Kotlin
claude "use @agent-kotlin-android-expert and create [feature]"
claude "use @agent-kotlin-android-expert and implement Jetpack Compose UI"
```

---

## 🔍 Quality Assurance

### Code Review
```bash
claude "use @agent-code-reviewer and review this code"
```

### Performance
```bash
claude "use @agent-performance-optimizer and find bottlenecks"
```

### Documentation
```bash
claude "use @agent-documentation-specialist and create README"
claude "use @agent-documentation-specialist and document API endpoints"
```

---

## 🌐 Agents Universais (quando não tem especialista)
```bash
# Backend genérico
claude "use @agent-backend-developer and [tarefa]"

# Frontend genérico
claude "use @agent-frontend-developer and [tarefa]"

# API genérica
claude "use @agent-api-architect and design API structure"
```

---

## 💡 Dicas Pro

### Multi-Agent Workflow
```bash
# 1. Analisar stack
claude "use @agent-project-analyst and detect stack"

# 2. Configurar time
claude "use @agent-team-configurator and setup optimal team"

# 3. Desenvolver com orchestrator
claude "use @agent-tech-lead-orchestrator and build feature"

# 4. Review
claude "use @agent-code-reviewer and review changes"

# 5. Otimizar
claude "use @agent-performance-optimizer and optimize"

# 6. Documentar
claude "use @agent-documentation-specialist and document"
```

### Comandos Combinados
```bash
# Criar feature E revisar
claude "use @agent-tech-lead-orchestrator to build authentication, then use @agent-code-reviewer to review it"

# Otimizar E documentar
claude "use @agent-performance-optimizer to find bottlenecks, then use @agent-documentation-specialist to document the improvements"
```

---

## 📂 Estrutura de Pastas Recomendada
```
C:\Users\dcpagotto\Documents\Projetos\
├── meu-projeto-1/
│   └── CLAUDE.md              # Configuração do time de AI
├── meu-projeto-2/
│   └── CLAUDE.md
└── Claude-Dev-Agents/         # Repositório dos agents
    ├── agents/                # Symlinked para ~/.claude/agents/
    ├── INSTALACAO-COMPLETA.md
    └── ATALHOS-RAPIDOS.md     # Este arquivo
```

---

## 🆘 Troubleshooting

### Agent não encontrado
```bash
# Verificar agents disponíveis
claude /agents

# Se não aparecer, verificar symlink
Get-Item "$env:USERPROFILE\.claude\agents\awesome-claude-agents"
```

### Atualizar agents
```bash
cd C:\Users\dcpagotto\Documents\Projetos\Claude-Dev-Agents
git pull origin main
```

### Recriar symlink
```powershell
# Remover symlink antigo
Remove-Item "$env:USERPROFILE\.claude\agents\awesome-claude-agents"

# Criar novo
cmd /c mklink /D "$env:USERPROFILE\.claude\agents\awesome-claude-agents" "C:\Users\dcpagotto\Documents\Projetos\Claude-Dev-Agents\agents"
```

---

✅ **Salve este arquivo como referência rápida!**
