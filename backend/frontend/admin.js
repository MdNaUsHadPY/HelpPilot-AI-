const API_BASE = "";
"use strict";

// Admin page: upload documents, list them, delete them. All text is inserted
// with textContent, never innerHTML, so file names cannot inject HTML (no XSS).

const MAX_FILE_BYTES = 10 * 1024 * 1024; // keep in sync with the server limit
const ALLOWED_EXTENSIONS = [".pdf", ".txt", ".docx"];
const PASSWORD_KEY = "supportBotAdminPassword"; // kept only for this browser tab

const loginCard = document.getElementById("login-card");
const loginForm = document.getElementById("login-form");
const passwordInput = document.getElementById("password-input");
const loginError = document.getElementById("login-error");
const adminContent = document.getElementById("admin-content");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const progressEl = document.getElementById("progress");
const progressBar = document.getElementById("progress-bar");
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const resultsEl = document.getElementById("results");
const docTable = document.getElementById("doc-table");
const docBody = document.getElementById("doc-body");
const emptyState = document.getElementById("empty-state");
const docsError = document.getElementById("docs-error");

let adminPassword = sessionStorage.getItem(PASSWORD_KEY) || "";

/** Raised when the server says the admin password is missing or wrong. */
class AuthError extends Error {}

// ---------------------------------------------------------------- helpers

function authHeaders() {
  return adminPassword ? { "X-Admin-Password": adminPassword } : {};
}

/** fetch() wrapper that adds the admin header and turns errors into exceptions. */
async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (response.status === 401) {
    throw new AuthError(data.error || "Incorrect admin password.");
  }
  if (!response.ok) {
    throw new Error(data.error || "Something went wrong. Please try again.");
  }
  return data;
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function showMessage(element, text) {
  element.textContent = text;
  element.hidden = !text;
}

// ------------------------------------------------------------ login screen

function showLogin(message = "") {
  adminContent.hidden = true;
  loginCard.hidden = false;
  showMessage(loginError, message);
  passwordInput.focus();
}

function showAdmin() {
  loginCard.hidden = true;
  adminContent.hidden = false;
}

async function openAdmin() {
  try {
    await loadDocuments();
    showAdmin();
  } catch (error) {
    if (error instanceof AuthError) {
      const hadPassword = adminPassword !== "";
      adminPassword = "";
      sessionStorage.removeItem(PASSWORD_KEY);
      showLogin(hadPassword ? error.message : "");
    } else {
      showAdmin();
      showMessage(docsError, error.message);
    }
  }
}

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  adminPassword = passwordInput.value;
  sessionStorage.setItem(PASSWORD_KEY, adminPassword);
  passwordInput.value = "";
  openAdmin();
});

// ---------------------------------------------------------- document table

function renderDocuments(documents) {
  docBody.replaceChildren();
  for (const doc of documents) {
    const row = document.createElement("tr");

    const nameCell = document.createElement("td");
    nameCell.className = "filename";
    nameCell.textContent = doc.filename;

    const chunkCell = document.createElement("td");
    chunkCell.className = "number";
    chunkCell.textContent = doc.chunks;

    const actionCell = document.createElement("td");
    actionCell.className = "actions";
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "btn btn-danger";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", () => deleteDocument(doc.filename));
    actionCell.append(deleteButton);

    row.append(nameCell, chunkCell, actionCell);
    docBody.append(row);
  }
  docTable.hidden = documents.length === 0;
  emptyState.hidden = documents.length !== 0;
}

async function loadDocuments() {
  const data = await api("/api/documents");
  showMessage(docsError, "");
  renderDocuments(data.documents);
}

async function deleteDocument(filename) {
  const confirmed = window.confirm(
    `Delete "${filename}"? The assistant will no longer use it to answer questions.`
  );
  if (!confirmed) {
    return;
  }
  try {
    await api(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    await loadDocuments();
  } catch (error) {
    if (error instanceof AuthError) {
      showLogin(error.message);
    } else {
      showMessage(docsError, error.message);
    }
  }
}

// ------------------------------------------------------------------ upload

function addResult(filename, text, isError) {
  const item = document.createElement("li");
  item.className = `result ${isError ? "error" : "ok"}`;
  const name = document.createElement("strong");
  name.textContent = filename;
  const detail = document.createElement("span");
  detail.textContent = text;
  item.append(name, detail);
  resultsEl.append(item);
}

/** Returns a problem description, or "" if the file looks fine to upload. */
function checkFile(file) {
  const lowerName = file.name.toLowerCase();
  if (!ALLOWED_EXTENSIONS.some((extension) => lowerName.endsWith(extension))) {
    return "Unsupported file type. Please upload a PDF, TXT or DOCX file.";
  }
  if (file.size === 0) {
    return "The file is empty.";
  }
  if (file.size > MAX_FILE_BYTES) {
    return "The file is too large (maximum 10 MB).";
  }
  return "";
}

function setProgress(fraction, label, indeterminate = false) {
  progressEl.hidden = false;
  progressBar.classList.toggle("indeterminate", indeterminate);
  progressFill.style.width = indeterminate ? "" : `${Math.round(fraction * 100)}%`;
  progressLabel.textContent = label;
}

/** Upload files with XMLHttpRequest, because fetch() cannot report upload progress. */
function sendFiles(files) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));

    const request = new XMLHttpRequest();
    request.open("POST", "/api/ingest");
    if (adminPassword) {
      request.setRequestHeader("X-Admin-Password", adminPassword);
    }
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const fraction = event.loaded / event.total;
        setProgress(fraction, `Uploading... ${Math.round(fraction * 100)}%`);
      }
    };
    request.upload.onload = () => {
      setProgress(1, "Reading and indexing your documents. This can take a moment...", true);
    };
    request.onerror = () => reject(new Error("The upload failed. Check your connection and try again."));
    request.onload = () => {
      let data = {};
      try {
        data = JSON.parse(request.responseText);
      } catch {
        // fall through to the generic error below
      }
      if (request.status === 401) {
        reject(new AuthError(data.error || "Incorrect admin password."));
      } else if (request.status !== 200) {
        reject(new Error(data.error || "The upload failed. Please try again."));
      } else {
        resolve(data.results);
      }
    };
    request.send(form);
  });
}

async function uploadFiles(files) {
  dropzone.classList.add("disabled");
  setProgress(0, "Uploading... 0%");
  try {
    const results = await sendFiles(files);
    for (const result of results) {
      if (result.error) {
        addResult(result.filename, result.error, true);
      } else {
        addResult(result.filename, `Added ${plural(result.chunks, "chunk")}.`, false);
      }
    }
    await loadDocuments();
  } catch (error) {
    if (error instanceof AuthError) {
      showLogin(error.message);
    } else {
      files.forEach((file) => addResult(file.name, error.message, true));
    }
  } finally {
    progressEl.hidden = true;
    dropzone.classList.remove("disabled");
  }
}

function handleFiles(fileList) {
  const files = Array.from(fileList);
  if (files.length === 0) {
    return;
  }
  resultsEl.replaceChildren();

  const validFiles = [];
  for (const file of files) {
    const problem = checkFile(file);
    if (problem) {
      addResult(file.name, problem, true);
    } else {
      validFiles.push(file);
    }
  }
  if (validFiles.length > 0) {
    uploadFiles(validFiles);
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  handleFiles(fileInput.files);
  fileInput.value = ""; // lets the same file be chosen again later
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  handleFiles(event.dataTransfer.files);
});

// Stop the browser from opening a file that is dropped outside the drop zone.
for (const eventName of ["dragover", "drop"]) {
  window.addEventListener(eventName, (event) => event.preventDefault());
}

// -------------------------------------------------------------------- start

async function init() {
  let protectedByPassword = false;
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    protectedByPassword = (await response.json()).admin_protected === true;
  } catch {
    // If health cannot be read, openAdmin() will show the real problem.
  }
  if (protectedByPassword && !adminPassword) {
    showLogin();
  } else {
    openAdmin();
  }
}

init();
