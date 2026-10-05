/* Optional local interactions. Complete static content remains readable without JS. */
'use strict';
(() => {
  document.documentElement.classList.add('js-enabled');
  const isChinese = document.documentElement.lang === 'zh-CN';
  const languageLink = document.querySelector('.language-link');
  const languageBase = languageLink ? languageLink.getAttribute('href') : '';
  const caseButtons = Array.from(document.querySelectorAll('[data-example-case]'));
  const caseKeys = new Set(caseButtons.map(button => button.dataset.exampleCase));

  function syncLanguage() {
    if (!languageLink) return;
    const target = new URL(languageBase, location.href);
    const current = new URL(location.href);
    const metric = current.searchParams.get('metric');
    const example = current.searchParams.get('case');
    if (metric === 'ap' || metric === 'f1') target.searchParams.set('metric', metric);
    else target.searchParams.delete('metric');
    if (example && caseKeys.has(example)) target.searchParams.set('case', example);
    else target.searchParams.delete('case');
    target.hash = location.hash;
    languageLink.setAttribute('href', languageBase + target.search + target.hash);
  }

  const metricButtons = Array.from(document.querySelectorAll('.metric-button[data-metric]'));
  const plots = Array.from(document.querySelectorAll('[data-metric-plot]'));
  function selectMetric(metric, updateURL) {
    if (!plots.length) return;
    const selected = metric === 'f1' ? 'f1' : 'ap';
    plots.forEach(plot => { plot.hidden = plot.dataset.metricPlot !== selected; });
    metricButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.metric === selected)));
    if (updateURL) {
      const target = new URL(location.href);
      target.searchParams.set('metric', selected);
      history.replaceState(null, '', target.href);
    }
    syncLanguage();
  }
  if (plots.length) {
    selectMetric(new URL(location.href).searchParams.get('metric'), false);
    metricButtons.forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault();
        selectMetric(button.dataset.metric, true);
      });
      button.addEventListener('keydown', event => {
        if (event.key === ' ' && button.tagName !== 'BUTTON') {
          event.preventDefault();
          button.click();
        }
      });
    });
  }

  document.querySelectorAll('.field-workbench').forEach((workbench, groupIndex) => {
    const buttons = Array.from(workbench.querySelectorAll('[data-example-case]'));
    const panels = Array.from(workbench.querySelectorAll('[data-example-panel]'));
    if (!buttons.length || !panels.length) return;
    function selectCase(key, updateURL) {
      const panel = panels.find(item => item.dataset.examplePanel === key);
      if (!panel) return;
      panels.forEach(item => { item.hidden = item !== panel; });
      buttons.forEach((button, index) => {
        const selected = button.dataset.exampleCase === key;
        const ownPanel = panels.find(item => item.dataset.examplePanel === button.dataset.exampleCase);
        if (ownPanel) {
          if (!ownPanel.id) ownPanel.id = 'example-panel-' + groupIndex + '-' + index;
          button.setAttribute('aria-controls', ownPanel.id);
        }
        if (button.getAttribute('role') === 'tab') {
          button.setAttribute('aria-selected', String(selected));
          button.tabIndex = selected ? 0 : -1;
        } else button.setAttribute('aria-pressed', String(selected));
      });
      if (updateURL) {
        const target = new URL(location.href);
        target.searchParams.set('case', key);
        history.replaceState(null, '', target.href);
      }
      syncLanguage();
    }
    const initial = new URL(location.href).searchParams.get('case');
    const initialButton = buttons.find(button => button.dataset.exampleCase === initial)
      || buttons.find(button => button.getAttribute('aria-pressed') === 'true' || button.getAttribute('aria-selected') === 'true')
      || buttons[0];
    selectCase(initialButton.dataset.exampleCase, false);
    buttons.forEach((button, index) => {
      button.addEventListener('click', event => {
        event.preventDefault();
        selectCase(button.dataset.exampleCase, true);
      });
      button.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % buttons.length;
        else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + buttons.length - 1) % buttons.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = buttons.length - 1;
        else if (event.key === ' ' && button.tagName !== 'BUTTON') {
          event.preventDefault();
          button.click();
        }
        if (next !== undefined) {
          event.preventDefault();
          buttons[next].focus();
          selectCase(buttons[next].dataset.exampleCase, true);
        }
      });
    });
    window.addEventListener('popstate', () => {
      const key = new URL(location.href).searchParams.get('case');
      const chosen = buttons.find(button => button.dataset.exampleCase === key) || buttons[0];
      selectCase(chosen.dataset.exampleCase, false);
    });
  });

  const tocLinks = Array.from(document.querySelectorAll('.report-sidebar a[href^="#"]'));
  const tocEntries = tocLinks.map(link => {
    let key;
    try { key = decodeURIComponent(link.getAttribute('href').slice(1)); }
    catch (_) { return null; }
    const section = document.getElementById(key);
    return section ? { link, section } : null;
  }).filter(Boolean);
  let scrollPending = false;
  function updateToc() {
    scrollPending = false;
    if (!tocEntries.length) return;
    const header = document.querySelector('.site-header');
    const offset = (header ? header.getBoundingClientRect().height : 70) + 35;
    let current = tocEntries[0];
    tocEntries.forEach(entry => {
      if (entry.section.getBoundingClientRect().top <= offset) current = entry;
    });
    tocEntries.forEach(entry => {
      const active = entry === current;
      entry.link.classList.toggle('is-active', active);
      if (active) entry.link.setAttribute('aria-current', 'location');
      else entry.link.removeAttribute('aria-current');
    });
  }
  function scheduleToc() {
    if (scrollPending) return;
    scrollPending = true;
    requestAnimationFrame(updateToc);
  }
  window.addEventListener('scroll', scheduleToc, { passive: true });
  window.addEventListener('resize', scheduleToc, { passive: true });
  window.addEventListener('hashchange', () => { syncLanguage(); scheduleToc(); });
  window.addEventListener('popstate', () => {
    selectMetric(new URL(location.href).searchParams.get('metric'), false);
    syncLanguage();
    scheduleToc();
  });
  updateToc();
  syncLanguage();

  document.querySelectorAll('[data-copy-target]').forEach(button => {
    button.hidden = false;
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      const status = document.getElementById(button.dataset.copyStatus);
      if (!target || !status) return;
      try {
        if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(target.textContent);
        status.textContent = isChinese ? '命令已复制。' : 'Commands copied.';
      } catch (_) {
        status.textContent = isChinese ? '无法访问剪贴板，请选中上方命令后手动复制。' : 'Clipboard access is unavailable. Select the commands above and copy them manually.';
      }
    });
  });
})();
