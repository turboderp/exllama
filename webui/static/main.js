var participants = [];
var chat_colors = [ "#343434",
                    "#383848",
                    "#374a48",
                    "#4a373d",
                    "#374a3d",
                    "#57503b",
                    "#3e2845",
                    "#452927" ]

var textbox_initial = "";

createGenSettings();
populate();

var currentNode = null;

// Send JSON packet to server, no response

function send(api, packet, ok_func = null, fail_func = null) {
    fetch(api, { method: "POST", headers: { "Content-Type": "application/json", },
        body: JSON.stringify(packet)})
        .then(response => response.json())
        .then(json => {
            if (json.hasOwnProperty("result")) {
                if (ok_func && json.result == "ok") ok_func();
                if (fail_func && json.result == "fail") fail_func();
            }
            else {
                console.log(json);
                throw new Error("Bad response from server.");
            }
        })
        .catch(error => {
            appendErrorMessage("Error: " + error);
            console.error('Error:', error);
        });
}

// Generator settings

function getTBNumber(id) {

    let num = 0;
    let v = document.getElementById(id).value;
    if (v != document.getElementById(id).dataset.text0) num = Number(v);
    return num;
}

function sendGenSettings() {

    let json = {};

    json.temperature = getTBNumber("sl_temperature_tb");
    json.top_p = getTBNumber("sl_topp_tb");
    json.min_p = getTBNumber("sl_minp_tb");
    json.top_k = getTBNumber("sl_topk_tb");
    json.typical = getTBNumber("sl_typical_tb");

    json.chunk_size = getTBNumber("sl_chunksize_tb");
    json.max_response_tokens = getTBNumber("sl_maxtokens_tb");
    json.gen_endnewline = document.querySelector("#cb_gen_endnewline input").checked;

    json.token_repetition_penalty_max = getTBNumber("sl_repp_penalty_tb");
    json.token_repetition_penalty_sustain = getTBNumber("sl_repp_sustain_tb");
    json.token_repetition_penalty_decay = getTBNumber("sl_repp_decay_tb");

    // console.log(json);
    send("/api/set_gen_settings", json);
}

function setSlider(id, value, override_max = null) {

    let slider = document.getElementById(id);
    if (override_max) slider.max = override_max;

    let tb = document.getElementById(id + "_tb");
    let decimals = slider.dataset.decimals;
    let mult = Math.pow(10, decimals);
    let tvalue = value * mult;
    slider.value = tvalue;

    if (tb.dataset.text0 != "" && slider.value == 0)
    {
        tb.value = tb.dataset.text0;
        tb.style.opacity = 0.4;
    }
    else
    {
        tb.value = value.toFixed(decimals);
        tb.style.opacity = 1;
    }
}

function sliderChanged(slider) {
    setSlider(slider.id, Number(slider.value) / Math.pow(10, slider.dataset.decimals));
}

function sliderChangedDone(slider) {
    sliderChanged(slider);
    sendGenSettings();
}

function sliderTBChangedDone(id, value) {

    let tb = document.getElementById(id);

    let nvalue = parseFloat(value);
    if (isNaN(nvalue)) {
        tb.value = textbox_initial;
        return;
    }

    let decimals = tb.dataset.decimals;
    let mult = Math.pow(10, decimals);
    let rvalue = Math.round(nvalue * mult) / mult;

    let actualmin = tb.dataset.actualmin;
    let actualmax = tb.dataset.actualmax;
    if (rvalue < actualmin / mult) rvalue = actualmin / mult;
    if (rvalue > actualmax / mult) rvalue = actualmax / mult;

    let sliderId = id.slice(0, -3);
    setSlider(sliderId, rvalue);

    sendGenSettings();
}

function prepSlider(id) {
    let tb = document.getElementById(id + "_tb");
    addTextboxEvents(tb, sliderTBChangedDone);
}

function createSlider(text, id, min, max, actualmax, decimals, text0 = null) {

    if (text) text0_str = "data-text0='" + (text0 ? text0 : "") + "' ";

    let newDiv = document.createElement('div');
    newDiv.className = 'custom-slider';
    newDiv.id = id + "_";
    newDiv.innerHTML = "<label for='" + id + "'>" + text + "</label>" +
                       "<input type='range' min='" + min + "' max='" + max + "' value='" + min + "' id='" + id + "' class = 'custom-slider-sl' data-decimals='" + decimals + "' oninput='sliderChanged(this)' onchange='sliderChangedDone(this)'>" +
                       "<input type='text' id='" + id + "_tb' class='custom-slider-tb' data-decimals='" + decimals + "' data-actualmax='" + actualmax + "' data-actualmin='" + min + "' " + text0_str + " />";

    return newDiv;
}

function createGenSettings() {

    // Sampling

    let settingsDiv = document.getElementById("gen_settings");
    settingsDiv.innerHTML = "";

    settingsDiv.appendChild(createSlider("Temperature", "sl_temperature", 1, 200, 100000, 2));
    settingsDiv.appendChild(createSlider("Top-K", "sl_topk", 0, 100, 32000, 0, "off"));
    settingsDiv.appendChild(createSlider("Top-P", "sl_topp", 0, 100, 100, 2, "off"));
    settingsDiv.appendChild(createSlider("Min-P", "sl_minp", 0, 99, 99, 2, "off"));
    settingsDiv.appendChild(createSlider("Typical", "sl_typical", 0, 100, 100, 2, "off"));

    prepSlider("sl_temperature");
    prepSlider("sl_topp");
    prepSlider("sl_minp");
    prepSlider("sl_topk");
    prepSlider("sl_typical");

    setSlider("sl_temperature", 1.00);
    setSlider("sl_topp", 0.65);
    setSlider("sl_minp", 0.00);
    setSlider("sl_topk", 40);
    setSlider("sl_typical", 0);

    // Stop conditions

    let stopDiv = document.getElementById("stop_settings");
    stopDiv.innerHTML = "";

    stopDiv.appendChild(createSlider("Max tokens", "sl_maxtokens", 32, 1024, 65536, 0));
    stopDiv.appendChild(createSlider("Chunk tokens", "sl_chunksize", 16, 1024, 65536, 0));

    prepSlider("sl_maxtokens");
    prepSlider("sl_chunksize");

    setSlider("sl_maxtokens", 512);
    setSlider("sl_chunksize", 128);

    document.getElementById("cb_gen_endnewline").addEventListener('change', sendGenSettings);

    // Repetition penalty

    let reppDiv = document.getElementById("repp_settings");
    reppDiv.innerHTML = "";

    reppDiv.appendChild(createSlider("Penalty", "sl_repp_penalty", 100, 200, 10000, 2));
    reppDiv.appendChild(createSlider("Sustain", "sl_repp_sustain", 0, 2048, 65536, 0));
    reppDiv.appendChild(createSlider("Decay", "sl_repp_decay", 0, 2048, 65536, 0));

    prepSlider("sl_repp_penalty");
    prepSlider("sl_repp_sustain");
    prepSlider("sl_repp_decay");

    setSlider("sl_repp_penalty", 1.15);
    setSlider("sl_repp_sustain", 1024);
    setSlider("sl_repp_decay", 512);

}

// Populate view

function populate() {
    fetch("/api/populate")
        .then(response => response.json())
        .then(json => {

            //console.log(json)
            var data = json;

            // Sessions list

            var current_session = data.current_session;
            var sessions = data.sessions;

            for (let i = 0; i < sessions.length; i++) {

                let newDiv = document.createElement('div');
                if (sessions[i] == current_session)
                    newDiv.className = 'session session_current no-select';
                else
                    newDiv.className = 'session no-select';

                newDiv.id = "session-button-" + i;
                newDiv.innerHTML = '<p>' + sessions[i] + '</p>';
                newDiv.dataset.sessionName = sessions[i];
                if (sessions[i] != current_session)
                {
                    newDiv.addEventListener('click', function() { setSession(this.dataset.sessionName); });
                    var icon = addIcon(newDiv, "delete-icon");
                    icon.addEventListener('click', function(event) { event.stopPropagation(); deleteSession(sessions[i], this); });
                }
                else
                {
                    var icon = addIcon(newDiv, "pencil-icon");
                    icon.addEventListener('click', function() { makeEditable(this.parentNode, renameSession) });
                }

                document.getElementById('sessions-column').appendChild(newDiv);
            }

            let newDiv = document.createElement('div');
            newDiv.className = 'session session_new no-select';
            newDiv.id = "session-button-new";
            newDiv.innerHTML = '<p>+ New...</p>';
            newDiv.addEventListener('click', function() { setSession("."); });
            document.getElementById('sessions-column').appendChild(newDiv);

            // Model info

            let tf_model_info = document.getElementById("tf_model_info")
            tf_model_info.value = data.model_info;

            let model_max_seq_lan = data.max_seq_len;

            // Fixed prompt

            let tf_fixed_prompt = document.getElementById("tf_fixed_prompt")
            tf_fixed_prompt.value = data.fixed_prompt;
            addTextboxEvents(tf_fixed_prompt, sendFixedPromptSettings);

            let cb_keep_fixed_prompt = document.querySelector("#cb_keep_fixed_prompt input")
            cb_keep_fixed_prompt.checked = data.keep_fixed_prompt;
            cb_keep_fixed_prompt.addEventListener('change', sendFixedPromptSettings);

            // Generator settings

            setSlider("sl_temperature", data.temperature);
            setSlider("sl_topp", data.top_p);
            setSlider("sl_minp", data.min_p);
            setSlider("sl_topk", data.top_k);
            setSlider("sl_typical", data.typical);

            // Stop conditions

            setSlider("sl_maxtokens", data.max_response_tokens);
            setSlider("sl_chunksize", data.chunk_size);

            let cb_gen_endnewline = document.querySelector("#cb_gen_endnewline input")
            cb_gen_endnewline.checked = data.break_on_newline;

            // Repetition penalty

            setSlider("sl_repp_penalty", data.token_repetition_penalty_max);
            setSlider("sl_repp_sustain", data.token_repetition_penalty_sustain, model_max_seq_lan);
            setSlider("sl_repp_decay", data.token_repetition_penalty_decay);

            // Participants

            participants = data.participants;
            updateParticipants();

            // Chat data

            let history = data.history;
            // console.log(history);

            for (let i = 0; i < history.length; i++) {

                let header = null;
                if (history[i].hasOwnProperty("author")) header = history[i].author;
                text = history[i].text;
                let idx = 0;
                if (history[i].hasOwnProperty("author_idx")) idx = history[i].author_idx;
                let uuid = history[i].uuid;

                newChatBlock(header, text, idx, uuid);
            }

            // Begin

            scrollToBottom();
            checkScroll();
            $("#chatbox").on('scroll', checkScroll);
            $("#user-input").focus();

        })
        .catch((error) => {
          console.error('Error:', error);
        });
}

// Edit chat block

var edit_uuid = null;
function editChatBlock(div, value) {

    send("/api/edit_block", { uuid: edit_uuid, text: value.trim() }, null, null);
}

// Delete chat block

function deleteChatBlock(uuid, div) {

    div.remove();
    send("/api/delete_block", { uuid: uuid }, null, null);

    $("#user-input").focus();
}

// Delete session

function deleteSession(session_name, div) {

    div.parentNode.remove();
    send("/api/delete_session", { session: session_name }, null, null);

}

// Rename session

function renameSession(div, value) {

    send("/api/rename_session", { new_name: value.trim() }, null, function() {
        div.firstElementChild.textContent = textbox_initial;
    } );

}

// Fixed prompt settings

function sendFixedPromptSettings() {

    let checked = document.querySelector("#cb_keep_fixed_prompt input").checked;
    let prompt = document.getElementById("tf_fixed_prompt").value;

    send("/api/set_fixed_prompt", { fixed_prompt: prompt, keep_fixed_prompt: checked });
}

// Participants list

function updateParticipants() {

    var container = document.getElementById("participants");
    container.innerHTML = "";

    if (participants.length == 0)
    {
        let newDiv = document.createElement('div');
        newDiv.className = 'textlabel';
        newDiv.innerHTML = 'No participants, running in free-form mode';
        document.getElementById('participants').appendChild(newDiv);
    }

    for (let i = 0; i < participants.length; i++) {

        let lname = "Bot " + i;
        if (participants.length == 2) lname = "Bot";
        if (i == 0) lname = "User";

        let newDiv = document.createElement('div');
        newDiv.className = 'custom-textbox';
        newDiv.id = "participant-" + i;
        newDiv.innerHTML = "<label for='tb_participant_" + i + "'>" + lname + "</label>" +
                           "<input type='text' id='tb_participant_" + i + "' />";

        if (i > 0) {

            let remDiv = document.createElement('div');
            remDiv.id = "remove_participant_" + i;
            remDiv.className = 'custom-textbox-rlabel no-select';
            remDiv.innerHTML = "✕ Remove";
            remDiv.addEventListener('click', function() { removeParticipant(remDiv.id); });
            newDiv.appendChild(remDiv);
        }

        document.getElementById('participants').appendChild(newDiv);

        let ctb = document.getElementById("tb_participant_" + i);
        ctb.value = participants[i];
        addTextboxEvents(ctb, saveParticipant);
    }

    if (participants.length < 8)  {

        let newDiv = document.createElement('div');
        newDiv.className = 'clickabletextlabel no-select';
        newDiv.innerHTML = '+ Add...';
        newDiv.addEventListener('click', function() { addParticipant(); });
        document.getElementById('participants').appendChild(newDiv);
    }
}

// Participants list

function addParticipant()
{
    if (participants.length == 0) {
        participants.push("");
        participants.push("Bot");
        updateParticipants();
        document.getElementById("tb_participant_0").focus();
    }

    if (participants.length > 0)
        for (let i = 0; i < participants.length; i++)
            if (participants[i].length == 0) {
                document.getElementById("tb_participant_" + i).focus();
                return;
            }

    participants.push("");
    updateParticipants("");
    document.getElementById("tb_participant_" + (participants.length - 1)).focus();
}

function removeParticipant(id)
{
    if (participants.length == 2) {
        participants = [];
        updateParticipants();
    } else {
        let num = parseInt(id.substring("remove-participant_".length));
        participants.splice(num, 1);
        updateParticipants();
    }

    send("/api/set_participants", { participants : participants });
}

function saveParticipant(id, value)
{
    if (value == "") return;
    let num = parseInt(id.substring("tb_participant_".length));
    if (num >= participants.length) participants.push("");
    participants[num] = value;

    send("/api/set_participants", { participants : participants });
}

// Textbox

function addTextboxEvents(ctb, onValueChange, multiline = true)
{
    ctb.addEventListener("focus",  function() {
        textbox_initial = this.value;
    });

    ctb.addEventListener("focusout", function() {
        if (textbox_initial !== this.value) {
            //console.log(this.id, this.value);
            onValueChange(this.id, this.value);
        }
    });

    ctb.addEventListener("keydown", function(event) {
        if (event.key === "Escape") {
            this.value = textbox_initial;
            this.blur();
            event.preventDefault();
        }
        if (event.key === "Enter" && !event.shiftKey) {
            this.blur();
        }
        if (event.key == "Enter" && !multiline)
        {
            this.blur();
            event.preventDefault();
        }
    });
}

// Make div editable

function lightenColor(color, percent) {
    var [r, g, b] = color.match(/\d+/g).map(Number); // Extract r, g, b values from string

    r = 255 * 0.04 + r * 0.96;
    g = 255 * 0.04 + g * 0.96;
    b = 255 * 0.04 + b * 0.96;

    // Ensure each channel is within the valid range
    r = Math.min(255, Math.max(0, Math.round(r)));
    g = Math.min(255, Math.max(0, Math.round(g)));
    b = Math.min(255, Math.max(0, Math.round(b)));

    return `rgb(${r}, ${g}, ${b})`;
}

function getTextWithLineBreaks(node) {
    let text = "";
    for (let child of node.childNodes) {
        if (child.nodeType === Node.TEXT_NODE) {
            text += child.textContent.trim();
        } else if (child.tagName === "BR") {
            text += "\n";
        }
    }
    return text;
}

function makeEditable(pdiv, onValueChange, multiline = false, mode = 0) {

    let dstyle = window.getComputedStyle(pdiv);
    let oldbackground = pdiv.style.backgroundColor;
    let oldborder = pdiv.style.borderColor;
    pdiv.style.backgroundColor = lightenColor(dstyle.backgroundColor);
    pdiv.style.borderColor = lightenColor(dstyle.borderColor);

    div = pdiv.firstElementChild;
    if (mode == 2) div = div.nextElementSibling.nextElementSibling;

    let text = getTextWithLineBreaks(div);
    let rect = div.getBoundingClientRect();
    let computedStyle = window.getComputedStyle(div);

    let textarea = document.createElement("textarea");

    textbox_initial = text;

    addTextboxEvents(textarea, onValueChange, multiline = multiline);

    // textarea.style.position = 'absolute';
    textarea.style.top = `${rect.top}px`;
    textarea.style.left = `${rect.left}px`;
    textarea.style.width = `${rect.width}px`;
    textarea.style.height = `${rect.height}px`;
    textarea.style.border = computedStyle.border;
    textarea.style.outline = computedStyle.outline;
    textarea.style.padding = computedStyle.padding;
    textarea.style.margin = computedStyle.margin;
    textarea.style.resize = computedStyle.resize;
    textarea.style.fontFamily = computedStyle.fontFamily;
    textarea.style.fontSize = computedStyle.fontSize;
    textarea.style.fontWeight = computedStyle.fontWeight;
    textarea.style.color = computedStyle.color;
    textarea.style.backgroundColor = computedStyle.backgroundColor;
    textarea.style.borderRadius = computedStyle.borderRadius;
    textarea.style.boxSizing = 'border-box';
    textarea.style.verticalAlign = computedStyle.verticalAlign;
    textarea.style.lineHeight = computedStyle.lineHeight;
    textarea.style.textAlign = computedStyle.textAlign;
    textarea.style.padding = computedStyle.padding;
    textarea.style.minHeight = `${rect.height}px`;
    textarea.rows = 1;

    if (mode == 1 || mode == 2) {
        textarea.style.width = "100%";

        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        });
    }

    textarea.value = text;

    var editMode = mode;
    textarea.addEventListener('blur', function () {

        div.innerHTML = "";
        if (editMode == 0) {
            let paragraph = document.createElement('p');
            paragraph.textContent = textarea.value;
            div.appendChild(paragraph);
        } else if (editMode == 1 || editMode == 2) {
            currentNode = document.createTextNode('');
            div.appendChild(currentNode);
            appendWithLineBreaks(currentNode, textarea.value.trim());
            edit_uuid = div.parentNode.dataset.uuid;
        }

        div.style.display = "";
        textarea.parentNode.removeChild(textarea);

        pdiv.style.backgroundColor = oldbackground;
        pdiv.style.borderColor = oldborder;
    });

    div.parentNode.insertBefore(textarea, div.nextSibling);
    div.style.display = "none";

    textarea.focus();
}

// Add icon

function addIcon(div, icon_id, offset = 0)
{
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttributeNS(null, "width", "30");
    svg.setAttributeNS(null, "height", "30");
    svg.style.position = "absolute";
    svg.style.right = (offset + 8).toString() + "px";
    svg.style.top = "8px";
    // svg.style.fill = "white";

    var use = document.createElementNS(ns, "use");
    use.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', "#" + icon_id);
    svg.appendChild(use);
    div.style.position = "relative";
    // var div = document.getElementById(divId);
    div.appendChild(svg);

    return svg;
}

// Expand/collapse control

$(document).ready(function() {
    $(".header").click(function() {
        $(this).parent().find(".contents").toggle();
        var arrow = $(this).find(".arrow");
        if (arrow.html() === "▼") arrow.html("▲");
        else arrow.html("▼");
    });
});

// Set session

function setSession(name) {
    send("/api/set_session", { session_name: name }, function() { location.reload(); });
}

// Handle text input

$(document).ready(function() {
    $("#chat-input-form").on('submit', function(e) {

        // Get user input

        e.preventDefault();
        var userInput = $("#user-input").val().trim();
        $("#user-input").val("").trigger('input');
        $("#user-input").prop("disabled", true);
        $("#user-input").attr('placeholder', '...');

        // Timeout

        let timeout = new Promise((resolve, reject) => {
            let id = setTimeout(() => {
                clearTimeout(id);
                reject('No response from server')
            }, 10000)
        })

        // Fetch request

        let fetchRequest = fetch("/api/userinput", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ user_input: userInput })
        });

        // Pass input to server and setup response reader

        Promise.race([fetchRequest, timeout])
            .then(response => {
                if (response.ok) {
                    return response.body;
                } else {
                    appendErrorMessage("Network response was not ok");
                    throw new Error("Network response was not ok.");
                }
            })
            .then(body => {
                processStream(body);
            })
            .catch(error => {
                appendErrorMessage("Error: " + error);
                console.error('Error:', error);
            });

    });
});

// Read stream

let data = '';

function processStream(stream) {
    let reader = stream.getReader();
    let decoder = new TextDecoder();
    let data = '';

    reader.read().then(function process({done, value}) {
        // console.log("Received chunk:", decoder.decode(value));
        if (done) {
            // console.log("Stream complete");
            $("#user-input").prop("disabled", false);
            $("#user-input").attr('placeholder', 'Type here...');
            $("#user-input").focus();
            return;
        }

        data += decoder.decode(value, {stream: true});
        let lines = data.split('\n');
        for (let i = 0; i < lines.length - 1; i++) {
            let json = null;
            try {
                json = JSON.parse(lines[i]);
            } catch(e) {
                console.error("Invalid JSON:", lines[i]);
                break;
            }
            processItem(json);
        }

        data = lines[lines.length - 1];

        reader.read().then(process);
    });
}

function appendWithLineBreaks(currentNode, text)
{
    var textChunks = text.split("\n");
    for (var i = 0; i < textChunks.length - 1; i++) {
        currentNode.textContent += textChunks[i];
        let br = document.createElement('br');
        currentNode.parentNode.appendChild(br);
        newNode = document.createTextNode('');
        currentNode.parentNode.appendChild(newNode);
        currentNode = newNode;
    }
    currentNode.textContent += textChunks[textChunks.length - 1];
    return currentNode;
}

function rewindWithLineBreaks(currentNode, chars)
{
    while (chars > 0) {

        if (currentNode.nodeName == "BR") {
            let parent = currentNode.parentNode;
            chars -= 1;
            parent.removeChild(currentNode);
            currentNode = parent.lastChild;
            continue;
        }

        if (currentNode.textContent.length >= chars) {
            currentNode.textContent = currentNode.textContent.slice(0, -chars);
            chars = 0;
            continue;
        }

        let parent = currentNode.parentNode;
        chars -= currentNode.textContent.length;
        parent.removeChild(currentNode);
        currentNode = parent.lastChild;
    }
    return currentNode;
}

function newChatBlock(header, text, idx, uuid)
{
    let xmode = 0;

    if (header) {
        $("#chatbox").append("<div class='box'><span class='name'>" + header + ":</span><br><span class='text'></span></div>");
        xmode = 2;
    } else {
        $("#chatbox").append("<div class='box'><span class='text'></span></div>");
        xmode = 1;
    }

    var div = document.getElementById("chatbox").lastElementChild;
    div.dataset.uuid = uuid;

    var icon = addIcon(div, "pencil-icon");
    icon.addEventListener('click', function() { makeEditable(this.parentNode, editChatBlock, true, xmode) });

    icon = addIcon(div, "delete-icon", 34);
    icon.addEventListener('click', function() { deleteChatBlock(uuid, div) });

    currentNode = document.createTextNode('');
    $(".text:last").append(currentNode);
    currentNode = appendWithLineBreaks(currentNode, text);
    $(".text:last").parent().css('background-color', chat_colors[idx]);

    return currentNode;
}

function processItem(data) {

    switch(data.cmd) {

        case "begin_stream": {
            scrollToBottom();
            break;
        }

        // Begin new text block

        case "begin_block": {
            let bottom = isAtBottom();

            let header = null;
            if (data.hasOwnProperty("author")) header = data.author;
            let init_text = "";
            if (data.hasOwnProperty("init_text")) init_text = data.init_text;
            let idx = 0;
            if (data.hasOwnProperty("author_idx")) idx = data.author_idx;
            let uuid = data.uuid;

            //console.log(header, init_text, idx);

            currentNode = newChatBlock(header, init_text, idx, uuid);

            if (bottom) scrollToBottom();
            break;
        }

        // Append incoming text

        case "append": {
            let bottom = isAtBottom();
            text = data.text;
            currentNode = appendWithLineBreaks(currentNode, text);
            if (bottom) scrollToBottom();
            break;
        }

        // Append incoming text

        case "rewind": {
            //console.log(data);
            let bottom = isAtBottom();
            chars = data.chars;
            currentNode = rewindWithLineBreaks(currentNode, chars);
            if (bottom) scrollToBottom();
            break;
        }

    }
}

// Error message

function appendErrorMessage(msg) {

    $("#chatbox").append("<p><span class='error'>" + msg + "</span></p>");
    scrollToBottom();
}

// Ignore shift-enter on input field to allow multiline input

$("#user-input").on('keydown', function(e) {

    if (e.which == 13 && !e.shiftKey) {
        e.preventDefault();
        $("#chat-input-form").submit();
    }
});

// Scrolling

function isAtBottom() {

    var $chatbox = $("#chatbox");
    return ($chatbox.scrollTop() + $chatbox.innerHeight() >= $chatbox[0].scrollHeight - 10);
}

function scrollToBottom() {

    var textareaHeight = $("#chat-input-form").outerHeight();
    $("#chatbox").scrollTop($("#chatbox")[0].scrollHeight - textareaHeight);
    checkScroll();
}

// Adjust height of input text field to fit current contents

$("#user-input").on('input', function() {

    this.style.height = 'inherit';

    let bottom = isAtBottom();

    let computed = window.getComputedStyle(this);
    let height = this.scrollHeight
        - parseInt(computed.getPropertyValue('border-top-width'), 10)
        - parseInt(computed.getPropertyValue('padding-top'), 10)
        - parseInt(computed.getPropertyValue('padding-bottom'), 10)
        - parseInt(computed.getPropertyValue('border-bottom-width'), 10);

    this.style.height = height + 'px';
    if (bottom) checkScroll();
});

// Update gradient overlays top and bottom when chat log is longer than view

function updateGradientPosition() {

    var $textarea = $("#chat-input-form");
    var $gradient_bot = $("#chat-gradient-bot");
    var textareaHeight = $textarea.outerHeight();

    $gradient_bot.css('bottom', (textareaHeight + 10) + 'px');
}

function checkScroll() {

    var $chatbox = $("#chatbox");
    var $gradient_bot = $("#chat-gradient-bot");
    var $gradient_top = $("#chat-gradient-top");

    if ($chatbox.scrollTop() < 2) $gradient_top.hide();
    else $gradient_top.show();

    if ($chatbox.scrollTop() + $chatbox.innerHeight() >= $chatbox[0].scrollHeight - 10) $gradient_bot.hide();
    else $gradient_bot.show();

    updateGradientPosition();
}
