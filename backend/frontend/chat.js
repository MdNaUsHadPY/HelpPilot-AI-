"use strict";

// Customer chat page. All text is inserted with textContent, never innerHTML,
// so user and bot messages can never inject HTML or scripts (no XSS).

const WELCOME_MESSAGE =
  "Hello! I'm the virtual support assistant. Ask me a question and I'll do my best to help.";
const CONNECTION_ERROR =
  "Sorry, I couldn't reach the support service. Please try again in a moment.";
const MAX_HISTORY_SENT = 10; // the server only uses the most recent few

const messagesEl = document.getElementById("messages");
const examplesEl = document.getElementById("examples");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");

// Conversation so far, as [{role: "user" | "assistant", content: "..."}].
const history = [];
let busy = false;

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/** Add a chat bubble. Returns the message element. */
function addMessage(role, text, { sources = [], isError = false } = {}) {
  const message = document.createElement("div");
  message.className = `message ${role}${isError ? " error" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  message.append(bubble);

  if (sources.length > 0) {
    const tag = document.createElement("div");
    tag.className = "sources";
    tag.textContent = `Sources: ${sources.join(", ")}`;
    message.append(tag);
  }

  messagesEl.append(message);
  scrollToBottom();
  return message;
}

/** Show the animated "bot is typing" bubble. Returns it so it can be removed. */
function showTyping() {
  const message = document.createElement("div");
  message.className = "message bot";
  const bubble = document.createElement("div");
  bubble.className = "bubble typing";
  bubble.setAttribute("aria-label", "The assistant is typing");
  for (let i = 0; i < 3; i += 1) {
    bubble.append(document.createElement("span"));
  }
  message.append(bubble);
  messagesEl.append(message);
  scrollToBottom();
  return message;
}

function setBusy(isBusy) {
  busy = isBusy;
  inputEl.disabled = isBusy;
  sendButton.disabled = isBusy;
  examplesEl.querySelectorAll("button").forEach((chip) => {
    chip.disabled = isBusy;
  });
}

async function sendMessage(text) {
  const question = text.trim();
  if (!question || busy) {
    return;
  }

  examplesEl.hidden = true;
  addMessage("user", question);
  inputEl.value = "";
  setBusy(true);
  const typing = showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, history: history.slice(-MAX_HISTORY_SENT) }),
    });
    const data = await response.json().catch(() => ({}));
    typing.remove();

    if (!response.ok) {
      addMessage("bot", data.error || CONNECTION_ERROR, { isError: true });
      return;
    }
    addMessage("bot", data.answer, { sources: data.sources || [] });
    history.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
  } catch {
    typing.remove();
    addMessage("bot", CONNECTION_ERROR, { isError: true });
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault(); // Enter and the Send button both land here
  sendMessage(inputEl.value);
});

examplesEl.querySelectorAll("button").forEach((chip) => {
  chip.addEventListener("click", () => sendMessage(chip.textContent));
});

addMessage("bot", WELCOME_MESSAGE);
