/// <reference types="node" />
import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

const docsBase = "/learn-world-model/";
const brandLogo = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23D95C41" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12.1" y1="11.9" x2="18.9" y2="8.2" /><line x1="12.1" y1="12.1" x2="20.3" y2="12.9" /><line x1="12.2" y1="12.4" x2="16.6" y2="19.1" /><line x1="11.8" y1="12.4" x2="7.3" y2="19.2" /><line x1="11.9" y1="12.1" x2="3.7" y2="13.3" /><line x1="11.8" y1="11.7" x2="7.8" y2="4.4" /></svg>';
const githubRepoTreeLink = "https://github.com/zhimin-z/learn-world-model/tree/main";

const zhWorldModelItems = [
  { text: "欢迎", link: "/zh/" },
  {
    text: "第一讲：内部仿真与历史脉络",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-01-internal-simulation/" },
      { text: "思想基石", link: "/zh/lectures/lecture-01-internal-simulation/01-foundations" },
      { text: "世界模型是什么：渲染、模拟与规划", link: "/zh/lectures/lecture-01-internal-simulation/02-what-is-a-world-model" },
      { text: "四个时代", link: "/zh/lectures/lecture-01-internal-simulation/03-four-eras" },
      { text: "世界模型的价值与时代背景", link: "/zh/lectures/lecture-01-internal-simulation/04-why-now" },
      { text: "课程路线图", link: "/zh/lectures/lecture-01-internal-simulation/05-roadmap" },
    ]
  },
  {
    text: "第二讲：观测编码与潜在动力学",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-02-encode-and-dynamics/" },
      { text: "Part A：观测编码", link: "/zh/lectures/lecture-02-encode-and-dynamics/01-encoding" },
      { text: "Part B：GRU、MDN-RNN、RSSM", link: "/zh/lectures/lecture-02-encode-and-dynamics/02-dynamics" },
      { text: "Part B（续）：Dreamer 系列迭代", link: "/zh/lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series" },
    ]
  },
  {
    text: "第三讲：架构模式、学习范式与规划",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-03-architecture-patterns/" },
      { text: "Part A：RNN、Transformer、Diffusion", link: "/zh/lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion" },
      { text: "Part A（续）：JEPA 与 RWM", link: "/zh/lectures/lecture-03-architecture-patterns/02-architectures-jepa-rwm-wam" },
      { text: "Part A（续二）：Genie、WAM 与选型", link: "/zh/lectures/lecture-03-architecture-patterns/03-architectures-genie-wam" },
      { text: "Part B：CEM-MPC 与 Actor-Critic", link: "/zh/lectures/lecture-03-architecture-patterns/03-planning-cem-ac" },
      { text: "Part B（续）：TD-MPC 与规划对比", link: "/zh/lectures/lecture-03-architecture-patterns/04-planning-tdmpc" },
      { text: "案例研究：LS-Imagine", link: "/zh/lectures/lecture-03-architecture-patterns/05-case-study-ls-imagine" },
    ]
  },
  {
    text: "第四讲：按模型评估",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-04-evaluation-by-model/" },
      { text: "Dreamer 指标", link: "/zh/lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer-muzero" },
      { text: "MuZero 指标", link: "/zh/lectures/lecture-04-evaluation-by-model/02-model-metrics-muzero" },
      { text: "TD-MPC 指标", link: "/zh/lectures/lecture-04-evaluation-by-model/02-model-metrics-tdmpc" },
      { text: "STORM 指标", link: "/zh/lectures/lecture-04-evaluation-by-model/03-storm-diffusion-drift" },
      { text: "扩散模型与时程漂移", link: "/zh/lectures/lecture-04-evaluation-by-model/04-diffusion-drift" },
      { text: "真实部署评测", link: "/zh/lectures/lecture-04-evaluation-by-model/04-deployment-metrics" },
      { text: "七个常见坑与落地策略", link: "/zh/lectures/lecture-04-evaluation-by-model/05-deployment-pitfalls" },
      { text: "总结、落地策略与展望", link: "/zh/lectures/lecture-04-evaluation-by-model/06-summary" },
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
  { text: "P01：训练 VAE 编码器", link: "/zh/projects/project-01-vae-encoder/" },
  { text: "P02：构建 RSSM 动力学模型", link: "/zh/projects/project-02-latent-dynamics/" },
  { text: "P03：训练 Dreamer 智能体", link: "/zh/projects/project-03-dreamer-agent/" },
  { text: "P04：替换动力学骨干网络", link: "/zh/projects/project-04-transformer-backbone/" },
  { text: "P05：世界模型评估仪表盘", link: "/zh/projects/project-05-evaluation-dashboard/" },
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
      { text: "Four Eras", link: "/en/lectures/lecture-01-internal-simulation/03-four-eras" },
      { text: "Why Now", link: "/en/lectures/lecture-01-internal-simulation/04-why-now" },
      { text: "Course Roadmap", link: "/en/lectures/lecture-01-internal-simulation/05-roadmap" },
    ]
  },
  {
    text: "Lecture 02: Encoding & Latent Dynamics",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-02-encode-and-dynamics/" },
      { text: "Part A: Observation Encoding", link: "/en/lectures/lecture-02-encode-and-dynamics/01-encoding" },
      { text: "Part B: GRU, MDN-RNN & RSSM", link: "/en/lectures/lecture-02-encode-and-dynamics/02-dynamics" },
      { text: "Part B (cont): Dreamer Series", link: "/en/lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series" },
    ]
  },
  {
    text: "Lecture 03: Architecture, Paradigms & Planning",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-03-architecture-patterns/" },
      { text: "Part A: RNN, Transformer & Diffusion", link: "/en/lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion" },
      { text: "Part A (cont): JEPA & RWM", link: "/en/lectures/lecture-03-architecture-patterns/02-architectures-jepa-rwm-wam" },
      { text: "Part A (cont): Genie, WAM & Selection", link: "/en/lectures/lecture-03-architecture-patterns/03-architectures-genie-wam" },
      { text: "Part B: CEM-MPC & Actor-Critic", link: "/en/lectures/lecture-03-architecture-patterns/03-planning-cem-ac" },
      { text: "Part B (cont): TD-MPC & Comparison", link: "/en/lectures/lecture-03-architecture-patterns/04-planning-tdmpc" },
      { text: "Case Study: LS-Imagine", link: "/en/lectures/lecture-03-architecture-patterns/05-case-study-ls-imagine" },
    ]
  },
  {
    text: "Lecture 04: Evaluation by Model",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-04-evaluation-by-model/" },
      { text: "Dreamer Metrics", link: "/en/lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer-muzero" },
      { text: "MuZero Metrics", link: "/en/lectures/lecture-04-evaluation-by-model/02-model-metrics-muzero" },
      { text: "TD-MPC Metrics", link: "/en/lectures/lecture-04-evaluation-by-model/02-model-metrics-tdmpc" },
      { text: "STORM Metrics", link: "/en/lectures/lecture-04-evaluation-by-model/03-storm-diffusion-drift" },
      { text: "Diffusion & Horizon Drift", link: "/en/lectures/lecture-04-evaluation-by-model/04-diffusion-drift" },
      { text: "Real-World Deployment Metrics", link: "/en/lectures/lecture-04-evaluation-by-model/04-deployment-metrics" },
      { text: "Seven Pitfalls & Deployment Strategies", link: "/en/lectures/lecture-04-evaluation-by-model/05-deployment-pitfalls" },
      { text: "Summary & Outlook", link: "/en/lectures/lecture-04-evaluation-by-model/06-summary" },
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

const enProjectItems = [
  { text: "Welcome", link: "/en/projects/" },
  { text: "P01: Train a VAE Encoder", link: "/en/projects/project-01-vae-encoder/" },
  { text: "P02: Build an RSSM Dynamics Model", link: "/en/projects/project-02-latent-dynamics/" },
  { text: "P03: Train a Dreamer Agent", link: "/en/projects/project-03-dreamer-agent/" },
  { text: "P04: Swap the Dynamics Backbone", link: "/en/projects/project-04-transformer-backbone/" },
  { text: "P05: World Model Evaluation Dashboard", link: "/en/projects/project-05-evaluation-dashboard/" },
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
      }
    }
}));
