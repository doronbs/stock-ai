// StockAI service worker: app shell offline, API always from network.
const CACHE = "stockai-v5";
const SHELL = ["/", "/static/app.js", "/static/style.css", "/manifest.webmanifest",
  "/static/icons/icon-192.png", "/static/icons/apple-touch-icon.png",
  "https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => Promise.allSettled(SHELL.map(u => c.add(u)))));
  self.skipWaiting();
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith("/api/")) return;            // live data only
  e.respondWith(
    fetch(e.request).then(r => {
      const copy = r.clone();
      caches.open(CACHE).then(c => c.put(e.request, copy)).catch(() => {});
      return r;
    }).catch(() => caches.match(e.request))
  );
});
self.addEventListener("notificationclick", e => {
  e.notification.close();
  e.waitUntil(self.clients.matchAll({type: "window"}).then(cs => cs.length ? cs[0].focus() : self.clients.openWindow("/")));
});
