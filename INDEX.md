# Índice Completo do Projeto - Awesome Claude Agents

**Data de Indexação:** 2024  
**Versão do Projeto:** Experimental  
**Total de Agentes:** 38 especializados

---

## 📋 Sumário Executivo

Este projeto é uma coleção de **38 agentes especializados de IA** que estendem as capacidades do Claude Code através de orquestração inteligente e expertise em domínios específicos. Os agentes trabalham juntos como uma equipe de desenvolvimento, cada um com expertise específica e padrões de delegação.

### Características Principais
- **38 agentes especializados** organizados em 4 categorias
- **Sistema de orquestração** com tech-lead-orchestrator
- **Suporte multi-framework**: Django, Rails, Laravel, React, Vue
- **Agentes de deploy**: Docker e Kubernetes para containerização e orquestração
- **Agentes universais** para fallback quando não há especialista
- **Agentes core** para qualidade, performance e documentação

---

## 📁 Estrutura de Diretórios

```
Claude-Dev-Agents/
├── agents/                          # Todos os agentes especializados
│   ├── core/                        # Agentes fundamentais (4 agentes)
│   ├── orchestrators/               # Orquestradores (3 agentes)
│   ├── specialized/                  # Agentes por framework (27 agentes)
│   │   ├── django/                  # Django (3 agentes)
│   │   ├── deploy/                  # Deploy (2 agentes)
│   │   ├── laravel/                 # Laravel (2 agentes)
│   │   ├── python/                  # Python geral (10 agentes)
│   │   ├── rails/                   # Rails (3 agentes)
│   │   ├── react/                   # React (2 agentes)
│   │   └── vue/                     # Vue (3 agentes)
│   └── universal/                   # Agentes universais (4 agentes)
├── docs/                            # Documentação do projeto
│   ├── best-practices.md            # Melhores práticas para criar agentes
│   ├── creating-agents.md           # Guia de criação de agentes
│   └── dependencies.md              # Dependências opcionais (Context7 MCP)
├── CLAUDE.md                        # Configuração principal para Claude Code
├── CONTRIBUTING.md                  # Guia de contribuição
├── LICENSE                          # Licença MIT
├── examples/                        # Exemplos de workflows multi-agente
├── tests/                           # Scripts de validação
└── README.md                        # Documentação principal do projeto
```

---

## 🎭 Categorias de Agentes

### 1. Orchestrators (3 agentes)
**Localização:** `agents/orchestrators/`

Agentes que coordenam e planejam tarefas complexas:

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **tech-lead-orchestrator** | `tech-lead-orchestrator.md` | Coordena projetos complexos através de workflow de 3 fases (Research → Planning → Execution) |
| **project-analyst** | `project-analyst.md` | Detecta stack tecnológico e habilita roteamento inteligente |
| **team-configurator** | `team-configurator.md` | Configura equipe de agentes e cria regras de roteamento em CLAUDE.md |

**Características:**
- Máximo 2 agentes em paralelo
- Retorna formato estruturado obrigatório
- Seleciona agentes baseado em contexto do sistema
- Usa apenas nomes exatos de agentes

---

### 2. Core Agents (4 agentes)
**Localização:** `agents/core/`

Agentes fundamentais para qualidade e análise:

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **code-archaeologist** | `code-archaeologist.md` | Explora, documenta e analisa codebases desconhecidos ou legados |
| **code-reviewer** | `code-reviewer.md` | Revisões rigorosas com foco em segurança, com relatórios marcados por severidade |
| **performance-optimizer** | `performance-optimizer.md` | Identifica gargalos e aplica otimizações para sistemas escaláveis |
| **documentation-specialist** | `documentation-specialist.md` | Cria READMEs, especificações de API e documentação técnica |

**Características:**
- Suportam todas as stacks tecnológicas
- Focam em preocupações transversais (cross-cutting)
- Retornam relatórios estruturados
- Podem delegar para especialistas quando necessário

---

### 3. Specialized Agents (27 agentes)
**Localização:** `agents/specialized/`

Agentes especializados por framework/tecnologia:

#### 3.1 Django (3 agentes)
**Localização:** `agents/specialized/django/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **django-backend-expert** | `django-backend-expert.md` | Desenvolvimento completo de backend Django (models, views, services) |
| **django-api-developer** | `django-api-developer.md` | APIs REST e GraphQL com Django REST Framework |
| **django-orm-expert** | `django-orm-expert.md` | Otimização de queries e performance de banco de dados |

**Expertise:**
- Django ORM, migrations, admin customization
- Django REST Framework, GraphQL
- Channels (WebSockets), Celery
- Query optimization, caching

#### 3.2 Laravel (2 agentes)
**Localização:** `agents/specialized/laravel/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **laravel-backend-expert** | `laravel-backend-expert.md` | Desenvolvimento Laravel completo com MVC, services e padrões Eloquent |
| **laravel-eloquent-expert** | `laravel-eloquent-expert.md` | Otimização avançada de ORM, queries complexas e performance de banco |

#### 3.3 Rails (3 agentes)
**Localização:** `agents/specialized/rails/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **rails-backend-expert** | `rails-backend-expert.md` | Desenvolvimento full-stack Rails seguindo convenções |
| **rails-api-developer** | `rails-api-developer.md` | APIs RESTful e GraphQL com padrões Rails |
| **rails-activerecord-expert** | `rails-activerecord-expert.md` | Queries complexas e otimização de banco de dados |

#### 3.4 React (2 agentes)
**Localização:** `agents/specialized/react/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **react-component-architect** | `react-component-architect.md` | Padrões modernos React, hooks e design de componentes |
| **react-nextjs-expert** | `react-nextjs-expert.md` | SSR, SSG, ISR e aplicações full-stack Next.js |

#### 3.5 Vue (3 agentes)
**Localização:** `agents/specialized/vue/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **vue-component-architect** | `vue-component-architect.md` | Vue 3 Composition API e padrões de componentes |
| **vue-nuxt-expert** | `vue-nuxt-expert.md` | SSR, SSG e aplicações full-stack Nuxt |
| **vue-state-manager** | `vue-state-manager.md` | Arquitetura de estado com Pinia e Vuex |

#### 3.6 Python (10 agentes)
**Localização:** `agents/specialized/python/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **python-expert** | `python-expert.md` | Core Python 3.12+ development, type hints, project architecture |
| **python-async-expert** | `python-async-expert.md` | Asyncio, Celery, event-driven architecture |
| **python-api-expert** | `python-api-expert.md` | Flask APIs, JWT/RBAC auth, RESTful patterns |
| **fastapi-expert** | `fastapi-expert.md` | High-performance async APIs with FastAPI and Pydantic V2 |
| **ml-data-expert** | `ml-data-expert.md` | Machine Learning, data science, scikit-learn, TensorFlow, PyTorch |
| **testing-expert** | `testing-expert.md` | Python testing, pytest, test automation and quality assurance |
| **security-expert** | `security-expert.md` | Python security, cryptography, vulnerability assessment |
| **performance-expert** | `performance-expert.md` | Python performance optimization, profiling, concurrency |
| **devops-cicd-expert** | `devops-cicd-expert.md` | Python DevOps, CI/CD, deployment automation |
| **web-scraping-expert** | `web-scraping-expert.md` | Web scraping, data extraction, automation |

#### 3.7 Deploy (2 agentes)
**Localização:** `agents/specialized/deploy/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **docker-expert** | `docker-expert.md` | Especialista em Docker, containerização, Docker Compose, multi-stage builds e deployment remoto via SSH |
| **kubernetes-expert** | `kubernetes-expert.md` | Especialista em Kubernetes, cluster management, deployments, Helm charts e orquestração cloud-native |

**Expertise:**
- Docker: Dockerfile optimization, multi-stage builds, Docker Compose, image security
- Kubernetes: Pod orchestration, Services, Ingress, ConfigMaps, Secrets, HPA, Helm
- Remote deployment: SSH connectivity verification, credential management
- Container registries: Image building, pushing, and distribution

#### 3.8 Database (1 agente)
**Localização:** `agents/specialized/database/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **database-expert** | `database-expert.md` | Especialista em SQL, PostgreSQL, SQLite, MySQL, design de banco de dados, otimização de queries, migrações e administração de bancos de dados |

**Expertise:**
- SQL: PostgreSQL, SQLite, MySQL/MariaDB, queries otimizadas, window functions, CTEs
- Database Design: Normalização, schema design, indexing strategies, partitioning
- Performance: Query profiling, index optimization, connection pooling, caching
- Migrations: Schema migrations, data migrations, version control, zero-downtime deployments
- Advanced: Stored procedures, full-text search, JSON support, replication, high availability

#### 3.9 Android (1 agente)
**Localização:** `agents/specialized/android/`

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **kotlin-android-expert** | `kotlin-android-expert.md` | Especialista em Kotlin e desenvolvimento Android, incluindo Jetpack Compose, MVVM, Material Design, Android SDK e arquitetura mobile |

**Expertise:**
- Kotlin: Coroutines, Flow, sealed classes, null safety, functional programming
- Android SDK: Activities, Fragments, Jetpack Compose, View System, AndroidX libraries
- Architecture: MVVM, Clean Architecture, MVI, Repository pattern, Dependency Injection
- UI/UX: Material Design 3, Jetpack Compose, XML layouts, responsive design, accessibility
- Data: Room Database, DataStore, Retrofit, OkHttp, WorkManager
- Performance: Memory management, UI optimization, background processing, image loading

---

### 4. Universal Agents (4 agentes)
**Localização:** `agents/universal/`

Agentes framework-agnósticos (fallback quando não há especialista):

| Agente | Arquivo | Função Principal |
|--------|---------|------------------|
| **backend-developer** | `backend-developer.md` | Desenvolvimento backend poliglota em múltiplas linguagens e frameworks |
| **frontend-developer** | `frontend-developer.md` | Tecnologias web modernas e design responsivo para qualquer framework |
| **api-architect** | `api-architect.md` | Design RESTful, GraphQL e arquitetura de API framework-agnóstica |
| **tailwind-css-expert** | `tailwind-css-expert.md` | Estilização Tailwind CSS, desenvolvimento utility-first e componentes responsivos |

**Características:**
- Usados quando não há agente específico para o framework
- Fornecem soluções genéricas mas competentes
- Sempre disponíveis como fallback

---

## 📚 Documentação

### Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| **README.md** | Documentação principal do projeto, quick start, lista de agentes |
| **CLAUDE.md** | Configuração para Claude Code, padrões de orquestração, protocolo de roteamento |
| **CONTRIBUTING.md** | Guia de contribuição, padrões de qualidade, processo de PR |
| **LICENSE** | Licença MIT |

### Documentação Técnica (`docs/`)

| Arquivo | Conteúdo |
|---------|----------|
| **best-practices.md** | Playbook para criar agentes de alto impacto, convenções de arquivo, frontmatter obrigatório |
| **creating-agents.md** | Guia completo de criação de agentes, padrão XML, integração de agentes |
| **dependencies.md** | Dependências opcionais (Context7 MCP para documentação) |

---

## 🔧 Padrões e Convenções

### Estrutura de Arquivo de Agente

Todos os agentes seguem este formato:

```yaml
---
name: agent-name                    # kebab-case, único
description: |                      # Quando e por que usar
  Descrição clara com exemplos XML.
  Examples:
  - <example>
    Context: Quando usar
    user: "Exemplo de requisição"
    assistant: "Vou usar @agent-name..."
    <commentary>Por que foi selecionado</commentary>
  </example>
tools: Read, Write, Grep           # Opcional - omitir herda todas
---

# Nome do Agente

[System prompt com expertise, workflow, padrões...]
```

### Convenções de Nomenclatura

- **Formato:** `kebab-case` (minúsculas com hífens)
- **Especificidade:** Seja específico (`react-component-architect` não apenas `react-developer`)
- **Domínio:** Inclua o domínio (`api-architect`, `ui-specialist`)

### Localização de Agentes

| Tipo | Localização | Precedência |
|------|-------------|-------------|
| **Project agents** | `.claude/agents/` | Mais alta (dentro do repo) |
| **User agents** | `~/.claude/agents/` | Global (todos os projetos) |

**Regra de conflito:** Um agente de projeto sobrescreve um agente de usuário com o mesmo nome.

---

## 🔄 Padrão de Orquestração

### Protocolo de Roteamento de Agentes

**CRÍTICO:** Para tarefas complexas:

1. **SEMPRE começar com tech-lead-orchestrator** para qualquer tarefa multi-etapa
2. **SEGUIR o mapa de roteamento** retornado pelo tech-lead EXATAMENTE
3. **USAR APENAS os agentes** explicitamente recomendados pelo tech-lead
4. **NUNCA selecionar agentes independentemente** - tech-lead sabe quais agentes existem

### Workflow de 3 Fases

1. **Research Phase**: Tech-lead analisa requisitos e retorna descobertas estruturadas
2. **Approval Gate**: Agente principal apresenta descobertas e aguarda aprovação humana
3. **Planning Phase**: Agente principal cria tarefas com TodoWrite baseado nas recomendações
4. **Execution Phase**: Agente principal invoca especialistas sequencialmente com contexto filtrado

### Formato de Resposta Obrigatório (Tech-Lead)

```markdown
### Task Analysis
- [Resumo do projeto - 2-3 bullets]
- [Stack tecnológico detectado]

### SubAgent Assignments
Task 1: [descrição] → AGENT: @agent-[nome-exato]
Task 2: [descrição] → AGENT: @agent-[nome-exato]

### Execution Order
- **Parallel**: Tasks [X, Y] (max 2 at once)
- **Sequential**: Task A → Task B → Task C

### Available Agents for This Project
- [agent-name]: [justificativa de uma linha]

### Instructions to Main Agent
- Delegar tarefa 1 para [agent]
- Após tarefa 1, executar tarefas 2 e 3 em paralelo
```

---

## 🛠️ Configuração de Ferramentas

### Herança de Ferramentas

- **Omitir campo `tools`** = herda TODAS as ferramentas disponíveis
- **Especificar `tools`** = restringe a um conjunto específico (para segurança)

### Ferramentas Disponíveis

Quando `tools` é omitido, o agente herda:
- Todas as ferramentas built-in do Claude Code (Read, Write, Edit, MultiEdit, Bash, Grep, Glob, LS, etc.)
- WebFetch para acessar documentação e recursos web
- Qualquer ferramenta MCP (Model Context Protocol) de servidores conectados

### Quando Especificar Ferramentas

Apenas quando você quer **restringir** as capacidades do agente:

```yaml
---
name: code-reviewer
description: "Revisa código sem fazer alterações"
tools: Read, Grep, Glob, Bash  # Apenas ferramentas read-only para segurança
---
```

---

## 📊 Estatísticas do Projeto

### Contagem de Agentes por Categoria

- **Orchestrators:** 3 agentes
- **Core:** 4 agentes
- **Specialized:** 27 agentes
  - Android: 1 (kotlin-android-expert)
  - Database: 1 (database-expert)
  - Django: 3
  - Deploy: 2 (docker-expert, kubernetes-expert)
  - Laravel: 2
  - Rails: 3
  - React: 2
  - Vue: 3
  - Python: 10
- **Universal:** 4 agentes

**Total:** 38 agentes especializados

### Arquivos de Documentação

- **README.md:** Documentação principal
- **CLAUDE.md:** Configuração e padrões
- **CONTRIBUTING.md:** Guia de contribuição
- **docs/best-practices.md:** Melhores práticas
- **docs/creating-agents.md:** Guia de criação
- **docs/dependencies.md:** Dependências

**Total:** 6 arquivos de documentação

---

## 🎯 Casos de Uso Comuns

### 1. Desenvolvimento Full-Stack

```
tech-lead-orchestrator → project-analyst → 
django-backend-expert → django-api-developer → 
react-component-architect → code-reviewer
```

### 2. Otimização de Performance

```
code-archaeologist → performance-optimizer → 
django-orm-expert → code-reviewer
```

### 3. Análise de Código Legado

```
code-archaeologist → documentation-specialist → 
tech-lead-orchestrator → [agentes especializados conforme necessário]
```

### 4. Desenvolvimento de API

```
api-architect → django-api-developer → 
code-reviewer → documentation-specialist
```

### 5. Deploy e Containerização

```
docker-expert → kubernetes-expert → 
code-reviewer → documentation-specialist
```

---

## 🔍 Busca Rápida

### Por Framework

- **Django:** `django-backend-expert`, `django-api-developer`, `django-orm-expert`
- **Laravel:** `laravel-backend-expert`, `laravel-eloquent-expert`
- **Rails:** `rails-backend-expert`, `rails-api-developer`, `rails-activerecord-expert`
- **React:** `react-component-architect`, `react-nextjs-expert`
- **Vue:** `vue-component-architect`, `vue-nuxt-expert`, `vue-state-manager`

### Por Tipo de Tarefa

- **Orquestração:** `tech-lead-orchestrator`, `project-analyst`, `team-configurator`
- **Backend:** `django-backend-expert`, `laravel-backend-expert`, `rails-backend-expert`, `backend-developer`
- **Frontend:** `react-component-architect`, `vue-component-architect`, `frontend-developer`, `tailwind-css-expert`
- **API:** `django-api-developer`, `rails-api-developer`, `api-architect`
- **Database:** `database-expert`, `django-orm-expert`, `laravel-eloquent-expert`, `rails-activerecord-expert`
- **Mobile:** `kotlin-android-expert`
- **Deploy:** `docker-expert`, `kubernetes-expert`, `devops-cicd-expert`
- **Qualidade:** `code-reviewer`, `performance-optimizer`, `code-archaeologist`
- **Documentação:** `documentation-specialist`

### Por Stack Tecnológico

- **Python/Django:** `python-expert`, `python-async-expert`, `python-api-expert`, `fastapi-expert`, `ml-data-expert`, `django-backend-expert`, `django-api-developer`, `django-orm-expert`
- **PHP/Laravel:** `laravel-backend-expert`, `laravel-eloquent-expert`
- **Ruby/Rails:** `rails-backend-expert`, `rails-api-developer`, `rails-activerecord-expert`
- **JavaScript/React:** `react-component-architect`, `react-nextjs-expert`
- **JavaScript/Vue:** `vue-component-architect`, `vue-nuxt-expert`, `vue-state-manager`
- **Database/SQL:** `database-expert`, `django-orm-expert`, `laravel-eloquent-expert`, `rails-activerecord-expert`
- **Kotlin/Android:** `kotlin-android-expert`

---

## 📝 Notas Importantes

### Avisos

- ⚠️ **Projeto experimental e intensivo em tokens**
- ⚠️ Orquestração multi-agente pode consumir 10-50k tokens por feature complexa
- ⚠️ Use com cautela e monitore seu uso

### Requisitos

- Claude Code CLI instalado e autenticado
- Assinatura Claude (necessária para workflows intensivos de agentes)
- Diretório de projeto ativo com codebase
- Opcional: Context7 MCP para acesso aprimorado à documentação

### Licença

MIT License - Use livremente em seus projetos!

---

## 🔗 Referências Rápidas

- **GitHub:** https://github.com/vijaythecoder/awesome-claude-agents
- **Documentação:** `docs/creating-agents.md`
- **Melhores Práticas:** `docs/best-practices.md`
- **Contribuindo:** `CONTRIBUTING.md`

---

**Última Atualização:** 2024  
**Mantido por:** Awesome Claude Agents Contributors

