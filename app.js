const progressBar = document.getElementById("progressBar");

function updateProgress() {
  const root = document.documentElement;
  const scrollable = root.scrollHeight - root.clientHeight;
  const value = scrollable > 0 ? (root.scrollTop / scrollable) * 100 : 0;
  progressBar.style.width = `${Math.min(100, Math.max(0, value))}%`;
}

document.addEventListener("scroll", updateProgress, { passive: true });
updateProgress();

document.querySelectorAll(".quiz-list details").forEach((item) => {
  item.addEventListener("toggle", () => {
    if (!item.open) return;
    document.querySelectorAll(".quiz-list details").forEach((other) => {
      if (other !== item) other.open = false;
    });
  });
});

document.querySelectorAll("a[href]").forEach((link) => {
  const href = link.getAttribute("href");
  if (!href || href.startsWith("#") || href.startsWith("mailto:")) return;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
});