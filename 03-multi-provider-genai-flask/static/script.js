"use strict";

const form = document.getElementById("analysisForm");
const providerSelect = document.getElementById(
    "providerSelect"
);
const providerModel = document.getElementById(
    "providerModel"
);
const providerDescription = document.getElementById(
    "providerDescription"
);
const messageInput = document.getElementById(
    "messageInput"
);
const messagesContainer = document.getElementById(
    "messagesContainer"
);
const welcomeMessage = document.getElementById(
    "welcomeMessage"
);
const loadingIndicator = document.getElementById(
    "loadingIndicator"
);
const submitButton = document.getElementById(
    "submitButton"
);
const clearButton = document.getElementById(
    "clearButton"
);
const characterCount = document.getElementById(
    "characterCount"
);

const maxMessageLength =
    window.APP_CONFIG.maxMessageLength;

function getSelectedOption() {
    return providerSelect.options[
        providerSelect.selectedIndex
    ] || null;
}

function updateProviderDetails() {
    const option = getSelectedOption();

    if (!option || option.disabled) {
        providerModel.textContent =
            "No configured provider";
        providerDescription.textContent =
            "Add an API key to the .env file.";
        return;
    }

    providerModel.textContent =
        `Model: ${option.dataset.model}`;

    providerDescription.textContent =
        option.dataset.description;
}

function setLoading(isLoading) {
    loadingIndicator.classList.toggle(
        "hidden",
        !isLoading
    );

    providerSelect.disabled = isLoading;
    messageInput.disabled = isLoading;
    submitButton.disabled = isLoading;

    submitButton.textContent = isLoading
        ? "Analyzing..."
        : "Analyze message";
}

function hideWelcomeMessage() {
    if (welcomeMessage) {
        welcomeMessage.classList.add("hidden");
    }
}

function scrollToBottom() {
    messagesContainer.scrollTop =
        messagesContainer.scrollHeight;
}

function createCard(className) {
    const card = document.createElement("article");
    card.className = `message-card ${className}`;
    return card;
}

function createHeading(text) {
    const heading = document.createElement("div");
    heading.className = "message-heading";
    heading.textContent = text;
    return heading;
}

function createText(text, className = "message-content") {
    const element = document.createElement("p");
    element.className = className;
    element.textContent = text;
    return element;
}

function appendUserMessage(
    message,
    providerLabel,
    model
) {
    hideWelcomeMessage();

    const card = createCard("user-message");

    card.append(
        createHeading(
            `You · ${providerLabel} · ${model}`
        ),
        createText(message)
    );

    messagesContainer.appendChild(card);
    scrollToBottom();
}

function createDetailRow(label, value) {
    const row = document.createElement("div");
    row.className = "detail-row";

    const labelElement = document.createElement("span");
    labelElement.className = "detail-label";
    labelElement.textContent = label;

    const valueElement = document.createElement("span");
    valueElement.className = "detail-value";
    valueElement.textContent = String(value);

    row.append(labelElement, valueElement);
    return row;
}

function appendAssistantMessage(data) {
    const card = createCard("assistant-message");

    const details = document.createElement("div");
    details.className = "details-grid";

    details.append(
        createDetailRow("Summary", data.summary),
        createDetailRow(
            "Sentiment",
            `${data.sentiment} / 100`
        ),
        createDetailRow("Category", data.category),
        createDetailRow(
            "Recommended action",
            data.action
        )
    );

    card.append(
        createHeading(
            `${data.provider_label} · ${data.model} · ${data.duration_ms} ms`
        ),
        createText(
            data.response,
            "assistant-response"
        ),
        details
    );

    messagesContainer.appendChild(card);
    scrollToBottom();
}

function appendErrorMessage(message) {
    const card = createCard("error-message");

    card.append(
        createHeading("Request failed"),
        createText(message)
    );

    messagesContainer.appendChild(card);
    scrollToBottom();
}

async function requestAnalysis(
    provider,
    message
) {
    const response = await fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            provider,
            message,
        }),
    });

    const data = await response
        .json()
        .catch(() => ({
            error: "The server returned invalid JSON.",
        }));

    if (!response.ok) {
        throw new Error(
            data.error || "The request failed."
        );
    }

    return data;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const selectedOption = getSelectedOption();
    const message = messageInput.value.trim();

    if (!selectedOption || selectedOption.disabled) {
        appendErrorMessage(
            "Select a configured provider."
        );
        return;
    }

    if (!message) {
        messageInput.focus();
        return;
    }

    const provider = selectedOption.value;
    const providerLabel =
        selectedOption.textContent
            .replace("(API key required)", "")
            .trim();
    const model = selectedOption.dataset.model;

    appendUserMessage(
        message,
        providerLabel,
        model
    );

    setLoading(true);

    try {
        const result = await requestAnalysis(
            provider,
            message
        );

        appendAssistantMessage(result);

        messageInput.value = "";
        characterCount.textContent =
            `0 / ${maxMessageLength}`;

    } catch (error) {
        appendErrorMessage(
            error instanceof Error
                ? error.message
                : "An unexpected error occurred."
        );

    } finally {
        setLoading(false);
        updateProviderDetails();
        messageInput.focus();
    }
});

providerSelect.addEventListener(
    "change",
    updateProviderDetails
);

messageInput.addEventListener("input", () => {
    characterCount.textContent =
        `${messageInput.value.length} / ${maxMessageLength}`;
});

messageInput.addEventListener(
    "keydown",
    (event) => {
        if (
            event.key === "Enter"
            && !event.shiftKey
        ) {
            event.preventDefault();
            form.requestSubmit();
        }
    }
);

clearButton.addEventListener("click", () => {
    messagesContainer
        .querySelectorAll(".message-card")
        .forEach((card) => card.remove());

    if (welcomeMessage) {
        welcomeMessage.classList.remove("hidden");
    }

    messageInput.value = "";
    characterCount.textContent =
        `0 / ${maxMessageLength}`;

    if (!messageInput.disabled) {
        messageInput.focus();
    }
});

updateProviderDetails();
