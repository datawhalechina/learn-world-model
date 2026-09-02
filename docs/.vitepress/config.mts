/// <reference types="node" />
import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

const docsBase = "/learn-world-model/";
const brandLogo = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23D95C41" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12.1" y1="11.9" x2="18.9" y2="8.2" /><line x1="12.1" y1="12.1" x2="20.3" y2="12.9" /><line x1="12.2" y1="12.4" x2="16.6" y2="19.1" /><line x1="11.8" y1="12.4" x2="7.3" y2="19.2" /><line x1="11.9" y1="12.1" x2="3.7" y2="13.3" /><line x1="11.8" y1="11.7" x2="7.8" y2="4.4" /></svg>';
const githubRepoTreeLink = "https://github.com/datawhalechina/learn-world-model";

const zhWorldModelItems = [
  { text: "欢迎", link: "/zh/" },
  {
    text: "第一讲：内部仿真与历史脉络",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-01-internal-simulation/" },
      { text: "思想基石", link: "/zh/lectures/lecture-01-internal-simulation/01-foundations" },
      { text: "世界模型是什么：渲染、模拟与规划", link: "/zh/lectures/lecture-01-internal-simulation/02-what-is-a-world-model" },
      { text: "世界模型的严格分类与四个时代", link: "/zh/lectures/lecture-01-internal-simulation/02-world-model-taxonomy" },
      { text: "世界模型的价值与时代背景", link: "/zh/lectures/lecture-01-internal-simulation/03-why-now" },
      { text: "课程路线图", link: "/zh/lectures/lecture-01-internal-simulation/04-roadmap" },
    ]
  },
  {
    text: "第二讲：观测编码与潜在动力学",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-02-encode-and-dynamics/" },
      { text: "观测编码", link: "/zh/lectures/lecture-02-encode-and-dynamics/01-encoding" },
      { text: "潜在动力学：GRU、MDN-RNN、RSSM", link: "/zh/lectures/lecture-02-encode-and-dynamics/02-dynamics" },
      { text: "Dreamer 系列迭代", link: "/zh/lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series" },
    ]
  },
  {
    text: "第三讲：架构模式、学习范式与规划",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-03-architecture-patterns/" },
      { text: "规划与控制：CEM-MPC、Actor-Critic", link: "/zh/lectures/lecture-03-architecture-patterns/05-planning-cem-ac" },
      { text: "规划与控制：TD-MPC 与机制对比", link: "/zh/lectures/lecture-03-architecture-patterns/06-planning-tdmpc" },
      { text: "骨干选择：RSSM、Transformer、扩散", link: "/zh/lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion" },
      { text: "选读：JEPA 与 RWM", link: "/zh/lectures/lecture-03-architecture-patterns/02-architectures-jepa-rwm-wam" },
      { text: "选读：空间 3D/4D 世界模型", link: "/zh/lectures/lecture-03-architecture-patterns/03-architectures-spatial-3d4d" },
      { text: "选读：Genie", link: "/zh/lectures/lecture-03-architecture-patterns/03-architectures-genie-wam" },
      { text: "选读：LoopWM、WAM 与选型", link: "/zh/lectures/lecture-03-architecture-patterns/04-architectures-loopwm-wam" },
      { text: "选读：七种系统接入模式", link: "/zh/lectures/lecture-03-architecture-patterns/07-system-integration-patterns" },
      { text: "选读案例：LS-Imagine", link: "/zh/lectures/lecture-03-architecture-patterns/07-case-study-ls-imagine" },
    ]
  },
  {
    text: "第四讲：世界模型诊断",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-04-evaluation-by-model/" },
      { text: "诊断框架：六个接口", link: "/zh/lectures/lecture-04-evaluation-by-model/00-diagnostic-framework" },
      { text: "表示与任务信号：Dreamer", link: "/zh/lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer" },
      { text: "潜在动力学与规划：TD-MPC", link: "/zh/lectures/lecture-04-evaluation-by-model/03-model-metrics-tdmpc" },
      { text: "价值与搜索：MuZero", link: "/zh/lectures/lecture-04-evaluation-by-model/02-model-metrics-muzero" },
      { text: "自回归 Rollout：STORM", link: "/zh/lectures/lecture-04-evaluation-by-model/04-storm-diffusion-drift" },
      { text: "物理一致性与时程漂移", link: "/zh/lectures/lecture-04-evaluation-by-model/05-diffusion-drift" },
      { text: "部署指标", link: "/zh/lectures/lecture-04-evaluation-by-model/06-deployment-metrics" },
      { text: "部署故障与策略", link: "/zh/lectures/lecture-04-evaluation-by-model/07-deployment-pitfalls" },
      { text: "诊断总结与展望", link: "/zh/lectures/lecture-04-evaluation-by-model/08-summary" },
    ]
  },
  {
    text: "第五讲：前沿思辨",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-05-frontier-debates/" },
      { text: "语言是鸦片与 Bitter Lesson", link: "/zh/lectures/lecture-05-frontier-debates/01-language-and-bitter-lesson" },
      { text: "世界模型与 LLM 的分工与收敛", link: "/zh/lectures/lecture-05-frontier-debates/02-agi-and-convergence" },
      { text: "数据从哪里来", link: "/zh/lectures/lecture-05-frontier-debates/03-data-and-future" },
      { text: "各路线核心赌注与收尾问题", link: "/zh/lectures/lecture-05-frontier-debates/04-bets-and-questions" },
      { text: "哲学后记：生成认知", link: "/zh/lectures/lecture-05-frontier-debates/05-enactive-cognition" },
    ]
  },
];

const zhProjectItems = [
  { text: "欢迎", link: "/zh/projects/" },
  { text: "P01：训练 VAE 编码器", link: "/zh/projects/p01_vae_encoder" },
  { text: "P02：构建 RSSM 动力学模型", link: "/zh/projects/p02_rssm_dynamics" },
  { text: "P03：训练 Dreamer 智能体", link: "/zh/projects/p03_dreamer_agent" },
  { text: "P04：替换动力学骨干网络", link: "/zh/projects/p04_transformer_backbone" },
  { text: "P05：世界模型评估仪表盘", link: "/zh/projects/p05_evaluation_dashboard" },
  { text: "P06：反事实的动作条件世界模型", link: "/zh/projects/p06_counterfactual_world_model" },
];

const enWorldModelItems = [
  { text: "Welcome", link: "/en/" },
  {
    text: "Lecture 01: Internal Simulation",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-01-internal-simulation/" },
      { text: "Foundations", link: "/en/lectures/lecture-01-internal-simulation/01-foundations" },
      { text: "What Is a World Model", link: "/en/lectures/lecture-01-internal-simulation/02-what-is-a-world-model" },
      { text: "World Model Taxonomy & Four Eras", link: "/en/lectures/lecture-01-internal-simulation/02-world-model-taxonomy" },
      { text: "Why Now", link: "/en/lectures/lecture-01-internal-simulation/03-why-now" },
      { text: "Course Roadmap", link: "/en/lectures/lecture-01-internal-simulation/04-roadmap" },
    ]
  },
  {
    text: "Lecture 02: Encoding & Latent Dynamics",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-02-encode-and-dynamics/" },
      { text: "Observation Encoding", link: "/en/lectures/lecture-02-encode-and-dynamics/01-encoding" },
      { text: "Latent Dynamics: GRU, MDN-RNN & RSSM", link: "/en/lectures/lecture-02-encode-and-dynamics/02-dynamics" },
      { text: "Dreamer Series Evolution", link: "/en/lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series" },
    ]
  },
  {
    text: "Lecture 03: Architecture, Paradigms & Planning",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-03-architecture-patterns/" },
      { text: "Planning & Control: CEM-MPC and Actor-Critic", link: "/en/lectures/lecture-03-architecture-patterns/05-planning-cem-ac" },
      { text: "Planning & Control: TD-MPC and Comparison", link: "/en/lectures/lecture-03-architecture-patterns/06-planning-tdmpc" },
      { text: "Backbone Selection: RSSM, Transformer, Diffusion", link: "/en/lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion" },
      { text: "Optional: JEPA and RWM", link: "/en/lectures/lecture-03-architecture-patterns/02-architectures-jepa-rwm-wam" },
      { text: "Optional: Spatial 3D/4D Models", link: "/en/lectures/lecture-03-architecture-patterns/03-architectures-spatial-3d4d" },
      { text: "Optional: Genie", link: "/en/lectures/lecture-03-architecture-patterns/03-architectures-genie-wam" },
      { text: "Optional: LoopWM, WAM and Selection", link: "/en/lectures/lecture-03-architecture-patterns/04-architectures-loopwm-wam" },
      { text: "Optional: Integration Patterns", link: "/en/lectures/lecture-03-architecture-patterns/07-system-integration-patterns" },
      { text: "Optional Case Study: LS-Imagine", link: "/en/lectures/lecture-03-architecture-patterns/07-case-study-ls-imagine" },
    ]
  },
  {
    text: "Lecture 04: Diagnosing World Models",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-04-evaluation-by-model/" },
      { text: "Framework: Six Interfaces", link: "/en/lectures/lecture-04-evaluation-by-model/00-diagnostic-framework" },
      { text: "Representation & Task Signal: Dreamer", link: "/en/lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer" },
      { text: "Latent Dynamics & Planning: TD-MPC", link: "/en/lectures/lecture-04-evaluation-by-model/03-model-metrics-tdmpc" },
      { text: "Value & Search: MuZero", link: "/en/lectures/lecture-04-evaluation-by-model/02-model-metrics-muzero" },
      { text: "Autoregressive Rollout: STORM", link: "/en/lectures/lecture-04-evaluation-by-model/04-storm-diffusion-drift" },
      { text: "Physical Consistency & Horizon Drift", link: "/en/lectures/lecture-04-evaluation-by-model/05-diffusion-drift" },
      { text: "Deployment Metrics", link: "/en/lectures/lecture-04-evaluation-by-model/06-deployment-metrics" },
      { text: "Deployment Failures & Strategies", link: "/en/lectures/lecture-04-evaluation-by-model/07-deployment-pitfalls" },
      { text: "Diagnostic Synthesis & Outlook", link: "/en/lectures/lecture-04-evaluation-by-model/08-summary" },
    ]
  },
  {
    text: "Lecture 05: Frontier Debates",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-05-frontier-debates/" },
      { text: "Language & Bitter Lesson", link: "/en/lectures/lecture-05-frontier-debates/01-language-and-bitter-lesson" },
      { text: "World Models vs LLMs: Convergence", link: "/en/lectures/lecture-05-frontier-debates/02-agi-and-convergence" },
      { text: "Where Does the Data Come From", link: "/en/lectures/lecture-05-frontier-debates/03-data-and-future" },
      { text: "Core Bets & Closing Questions", link: "/en/lectures/lecture-05-frontier-debates/04-bets-and-questions" },
      { text: "Philosophical Coda: Enactive Cognition", link: "/en/lectures/lecture-05-frontier-debates/05-enactive-cognition" },
    ]
  },
];

const koWorldModelItems = [
  { text: "환영합니다", link: "/ko/" },
  {
    text: "제1강: 내부 시뮬레이션과 역사적 맥락",
    collapsed: false,
    items: [
      { text: "개요", link: "/ko/lectures/lecture-01-internal-simulation/" },
      { text: "이론적 초석", link: "/ko/lectures/lecture-01-internal-simulation/01-foundations" },
      { text: "월드모델이란 무엇인가: 렌더링, 시뮬레이션, 계획", link: "/ko/lectures/lecture-01-internal-simulation/02-what-is-a-world-model" },
      { text: "월드모델의 엄격한 분류와 네 시대", link: "/ko/lectures/lecture-01-internal-simulation/02-world-model-taxonomy" },
      { text: "월드모델은 무엇을 해결하는가, 그리고 왜 지금인가", link: "/ko/lectures/lecture-01-internal-simulation/03-why-now" },
      { text: "강좌 로드맵", link: "/ko/lectures/lecture-01-internal-simulation/04-roadmap" },
    ]
  },
  {
    text: "제2강: 관측 인코딩과 잠재 동역학",
    collapsed: false,
    items: [
      { text: "개요", link: "/ko/lectures/lecture-02-encode-and-dynamics/" },
      { text: "Part A: 관측 인코딩", link: "/ko/lectures/lecture-02-encode-and-dynamics/01-encoding" },
      { text: "Part B: GRU, MDN-RNN, RSSM", link: "/ko/lectures/lecture-02-encode-and-dynamics/02-dynamics" },
      { text: "Part B(계속): Dreamer 시리즈", link: "/ko/lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series" },
    ]
  },
  {
    text: "제3강: 아키텍처 패턴, 학습 패러다임과 계획",
    collapsed: false,
    items: [
      { text: "개요", link: "/ko/lectures/lecture-03-architecture-patterns/" },
      { text: "계획과 제어: CEM-MPC, Actor-Critic", link: "/ko/lectures/lecture-03-architecture-patterns/05-planning-cem-ac" },
      { text: "계획과 제어: TD-MPC와 비교", link: "/ko/lectures/lecture-03-architecture-patterns/06-planning-tdmpc" },
      { text: "백본 선택: RSSM, Transformer, Diffusion", link: "/ko/lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion" },
      { text: "선택: JEPA와 RWM", link: "/ko/lectures/lecture-03-architecture-patterns/02-architectures-jepa-rwm-wam" },
      { text: "선택: 공간 3D/4D 모델", link: "/ko/lectures/lecture-03-architecture-patterns/03-architectures-spatial-3d4d" },
      { text: "선택: Genie", link: "/ko/lectures/lecture-03-architecture-patterns/03-architectures-genie-wam" },
      { text: "선택: LoopWM, WAM과 아키텍처 선택", link: "/ko/lectures/lecture-03-architecture-patterns/04-architectures-loopwm-wam" },
      { text: "선택: 일곱 가지 시스템 통합 패턴", link: "/ko/lectures/lecture-03-architecture-patterns/07-system-integration-patterns" },
      { text: "선택적 사례: LS-Imagine", link: "/ko/lectures/lecture-03-architecture-patterns/07-case-study-ls-imagine" },
    ]
  },
  {
    text: "제4강: 월드모델 진단하기",
    collapsed: false,
    items: [
      { text: "개요", link: "/ko/lectures/lecture-04-evaluation-by-model/" },
      { text: "진단 프레임워크: 여섯 인터페이스", link: "/ko/lectures/lecture-04-evaluation-by-model/00-diagnostic-framework" },
      { text: "표현과 과제 수행 신호: Dreamer", link: "/ko/lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer" },
      { text: "잠재 동역학과 계획: TD-MPC", link: "/ko/lectures/lecture-04-evaluation-by-model/03-model-metrics-tdmpc" },
      { text: "가치와 탐색: MuZero", link: "/ko/lectures/lecture-04-evaluation-by-model/02-model-metrics-muzero" },
      { text: "자기회귀 롤아웃: STORM", link: "/ko/lectures/lecture-04-evaluation-by-model/04-storm-diffusion-drift" },
      { text: "물리적 일관성과 호라이즌 드리프트", link: "/ko/lectures/lecture-04-evaluation-by-model/05-diffusion-drift" },
      { text: "배포 지표", link: "/ko/lectures/lecture-04-evaluation-by-model/06-deployment-metrics" },
      { text: "배포 실패와 전략", link: "/ko/lectures/lecture-04-evaluation-by-model/07-deployment-pitfalls" },
      { text: "진단 종합과 전망", link: "/ko/lectures/lecture-04-evaluation-by-model/08-summary" },
    ]
  },
  {
    text: "제5강: 최전선 논쟁",
    collapsed: false,
    items: [
      { text: "개요", link: "/ko/lectures/lecture-05-frontier-debates/" },
      { text: "언어라는 아편과 Bitter Lesson", link: "/ko/lectures/lecture-05-frontier-debates/01-language-and-bitter-lesson" },
      { text: "월드모델과 LLM의 분업과 수렴", link: "/ko/lectures/lecture-05-frontier-debates/02-agi-and-convergence" },
      { text: "데이터는 어디서 오는가", link: "/ko/lectures/lecture-05-frontier-debates/03-data-and-future" },
      { text: "각 노선의 핵심 베팅과 마무리 질문", link: "/ko/lectures/lecture-05-frontier-debates/04-bets-and-questions" },
      { text: "철학적 후기: 행화 인지", link: "/ko/lectures/lecture-05-frontier-debates/05-enactive-cognition" },
    ]
  },
];

const enProjectItems = [
  { text: "Welcome", link: "/en/projects/" },
  { text: "P01: Train a VAE Encoder", link: "/en/projects/p01_vae_encoder" },
  { text: "P02: Build an RSSM Dynamics Model", link: "/en/projects/p02_rssm_dynamics" },
  { text: "P03: Train a Dreamer Agent", link: "/en/projects/p03_dreamer_agent" },
  { text: "P04: Swap the Dynamics Backbone", link: "/en/projects/p04_transformer_backbone" },
  { text: "P05: World Model Evaluation Dashboard", link: "/en/projects/p05_evaluation_dashboard" },
  { text: "P06: Counterfactual Action-Conditioned World Model", link: "/en/projects/p06_counterfactual_world_model" },
];

const koProjectItems = [
  { text: "개요", link: "/ko/projects/" },
  { text: "P01: VAE 인코더 학습", link: "/ko/projects/p01_vae_encoder" },
  { text: "P02: RSSM 동역학 모델 구축", link: "/ko/projects/p02_rssm_dynamics" },
  { text: "P03: Dreamer 에이전트 학습", link: "/ko/projects/p03_dreamer_agent" },
  { text: "P04: 동역학 백본 교체", link: "/ko/projects/p04_transformer_backbone" },
  { text: "P05: 월드모델 평가 대시보드", link: "/ko/projects/p05_evaluation_dashboard" },
  { text: "P06: 동작 조건화 반사실적 월드모델", link: "/ko/projects/p06_counterfactual_world_model" },
];

export default withMermaid(
  defineConfig({
    base: docsBase,
    title: "Learn World Models",
    description:
      "A project-based curriculum on world models: from VAE encoders and latent dynamics to Dreamer, TD-MPC, STORM, and frontier debates on language vs physical grounding.",
    cleanUrls: true,
    srcExclude: ["temp/**"],
    ignoreDeadLinks: true,
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: brandLogo }]
    ],
    themeConfig: {
      logo: brandLogo,
      search: {
        provider: "local"
      },
      socialLinks: [{ icon: "github", link: githubRepoTreeLink }]
    },
    markdown: {
      theme: {
        light: 'github-light',
        dark: 'github-dark'
      },
      languages: ['bibtex'],
      math: true
    },
    mermaid: {
      theme: 'base',
      themeVariables: {
        primaryColor: '#F4F3EE',
        primaryBorderColor: '#D1D1D1',
        primaryTextColor: '#1A1A1A',
        lineColor: '#B3B3B3',
        fontFamily: 'Inter, sans-serif',
        fontSize: '18px'
      },
      flowchart: {
        nodeSpacing: 40,
        rankSpacing: 56,
        padding: 12
      }
    },
    locales: {
      root: {
        label: "简体中文",
        lang: "zh-CN",
        link: "/zh/",
        themeConfig: {
          nav: [
            { text: "讲义", link: "/zh/", activeMatch: '^/zh/(|lectures/)' },
            { text: "项目", link: zhProjectItems[0].link, activeMatch: '^/zh/projects/' },
          ],
          sidebar: {
            '/zh/projects/': [{ text: "项目", items: zhProjectItems }],
            '/zh/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-01-internal-simulation/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-02-encode-and-dynamics/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-03-architecture-patterns/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-04-evaluation-by-model/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-05-frontier-debates/': [{ text: "世界模型", items: zhWorldModelItems }],
          },
          outline: {
            level: [2, 3]
          },
          docFooter: {
            prev: "上一篇",
            next: "下一篇"
          },
          lastUpdated: {
            text: "最后更新于"
          },
          returnToTopLabel: "回到顶部",
          sidebarMenuLabel: "菜单",
          darkModeSwitchLabel: "主题",
          lightModeSwitchTitle: "切换到浅色模式",
          darkModeSwitchTitle: "切换到深色模式",
          socialLinks: [{ icon: "github", link: githubRepoTreeLink }]
        }
      },
      en: {
        label: "English",
        lang: "en",
        link: "/en/",
        themeConfig: {
          nav: [
            { text: "Lectures", link: "/en/", activeMatch: '^/en/(|lectures/)' },
            { text: "Projects", link: enProjectItems[0].link, activeMatch: '^/en/projects/' },
          ],
          sidebar: {
            '/en/projects/': [{ text: "Projects", items: enProjectItems }],
            '/en/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-01-internal-simulation/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-02-encode-and-dynamics/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-03-architecture-patterns/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-04-evaluation-by-model/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-05-frontier-debates/': [{ text: "World Models", items: enWorldModelItems }],
          },
          socialLinks: [{ icon: "github", link: githubRepoTreeLink }]
        }
      },
      ko: {
        label: "한국어",
        lang: "ko-KR",
        link: "/ko/",
        themeConfig: {
          nav: [
            { text: "강의", link: "/ko/", activeMatch: '^/ko/(|lectures/)' },
            { text: "프로젝트", link: koProjectItems[0].link, activeMatch: '^/ko/projects/' },
          ],
          sidebar: {
            '/ko/projects/': [{ text: "프로젝트", items: koProjectItems }],
            '/ko/': [{ text: "월드모델", items: koWorldModelItems }],
            '/ko/lectures/lecture-01-internal-simulation/': [{ text: "월드모델", items: koWorldModelItems }],
            '/ko/lectures/lecture-02-encode-and-dynamics/': [{ text: "월드모델", items: koWorldModelItems }],
            '/ko/lectures/lecture-03-architecture-patterns/': [{ text: "월드모델", items: koWorldModelItems }],
            '/ko/lectures/lecture-04-evaluation-by-model/': [{ text: "월드모델", items: koWorldModelItems }],
            '/ko/lectures/lecture-05-frontier-debates/': [{ text: "월드모델", items: koWorldModelItems }],
          },
          outline: {
            level: [2, 3]
          },
          docFooter: {
            prev: "이전 글",
            next: "다음 글"
          },
          lastUpdated: {
            text: "마지막 업데이트"
          },
          returnToTopLabel: "맨 위로",
          sidebarMenuLabel: "메뉴",
          darkModeSwitchLabel: "테마",
          lightModeSwitchTitle: "라이트 모드로 전환",
          darkModeSwitchTitle: "다크 모드로 전환",
          socialLinks: [{ icon: "github", link: githubRepoTreeLink }]
        }
      }
    }
}));
