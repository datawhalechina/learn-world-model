<template>
  <div class="not-found">
    <div class="not-found__content">
      <p class="not-found__emoji">🚧</p>
      <h1 class="not-found__title">Coming Soon</h1>
      <p class="not-found__desc">
        This page is still being written. Check back later or explore the lectures and projects already available in the sidebar.
      </p>
      <a class="not-found__link" href="/learn-world-model/">← Back to home</a>
    </div>
  </div>
</template>

<style scoped>
.not-found {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - var(--vp-nav-height));
  padding: 2rem;
}

.not-found__content {
  text-align: center;
  max-width: 480px;
}

.not-found__emoji {
  font-size: 3rem;
  margin: 0 0 1rem;
  line-height: 1;
}

.not-found__title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 1rem;
  color: var(--vp-c-text-1);
}

.not-found__desc {
  font-size: 1rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin: 0 0 2rem;
}

.not-found__link {
  display: inline-block;
  padding: 0.5rem 1.25rem;
  border-radius: 6px;
  background: var(--vp-c-brand-1);
  color: #fff;
  font-weight: 500;
  text-decoration: none;
  transition: background 0.2s;
}

.not-found__link:hover {
  background: var(--vp-c-brand-2);
}
</style>
