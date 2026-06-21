// State
let inputs = [0, 0];
let numHidden = 3;
let weightsIn = [];    // weightsIn[input_idx][hidden_idx]
let weightsOut = [];   // weightsOut[hidden_idx]
let biasesHidden = []; // biasesHidden[hidden_idx]
let biasOut = 0;

function initNetwork() {
    const inputCountEl = document.getElementById('hidden_count');
    numHidden = Math.max(1, Math.min(10, parseInt(inputCountEl.value) || 3));
    inputCountEl.value = numHidden;

    weightsIn = [new Array(numHidden).fill(0.0), new Array(numHidden).fill(0.0)];
    weightsOut = new Array(numHidden).fill(0.0);
    biasesHidden = new Array(numHidden).fill(0.0);
    biasOut = 0.0;

    renderNeurons();
    renderSliders();
    updateNetwork();
}

function toggleInput(index) {
    inputs[index] = inputs[index] === 0 ? 1 : 0;
    const btn = document.getElementById(`input_${index}`);
    btn.textContent = inputs[index];
    if (inputs[index] === 1) btn.classList.add('active');
    else btn.classList.remove('active');
    updateNetwork();
}

function renderNeurons() {
    const hiddenContainer = document.getElementById('hidden_layer_neurons');
    hiddenContainer.innerHTML = '';
    for (let i = 0; i < numHidden; i++) {
        const neuron = document.createElement('div');
        neuron.id = `hidden_${i}`;
        neuron.className = 'neuron';
        neuron.textContent = '0';
        hiddenContainer.appendChild(neuron);
    }
}

function renderSliders() {
    const inSlidersContainer = document.getElementById('input_weights_sliders');
    const hiddenBiasContainer = document.getElementById('hidden_biases_sliders');
    const outSlidersContainer = document.getElementById('hidden_weights_sliders');
    const outBiasContainer = document.getElementById('output_bias_slider');

    inSlidersContainer.innerHTML = '';
    hiddenBiasContainer.innerHTML = '';
    outSlidersContainer.innerHTML = '';
    outBiasContainer.innerHTML = '';

    for (let i = 0; i < 2; i++)
        for (let j = 0; j < numHidden; j++)
            inSlidersContainer.appendChild(createSliderMarkup(`w_in_${i}_${j}`, `W(In${i}→H${j})`, 0.0, (val) => { weightsIn[i][j] = val; }));

    for (let j = 0; j < numHidden; j++)
        hiddenBiasContainer.appendChild(createSliderMarkup(`b_hidden_${j}`, `B(H${j})`, 0.0, (val) => { biasesHidden[j] = val; }));

    for (let j = 0; j < numHidden; j++)
        outSlidersContainer.appendChild(createSliderMarkup(`w_out_${j}`, `W(H${j}→Out)`, 0.0, (val) => { weightsOut[j] = val; }));

    outBiasContainer.appendChild(createSliderMarkup(`b_out`, `B(Out)`, 0.0, (val) => { biasOut = val; }));
}

function createSliderMarkup(id, labelText, defaultVal, callback) {
    const wrapper = document.createElement('div');
    wrapper.className = 'slider-wrapper';

    const label = document.createElement('label');
    label.setAttribute('for', id);
    label.textContent = `${labelText}:`;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = id;
    slider.min = '-1';
    slider.max = '1';
    slider.step = '0.1';
    slider.value = defaultVal;

    const valDisplay = document.createElement('span');
    valDisplay.textContent = parseFloat(defaultVal).toFixed(1);

    slider.oninput = (e) => {
        const val = parseFloat(e.target.value);
        valDisplay.textContent = val.toFixed(1);
        callback(val);
        updateNetwork();
    };

    wrapper.appendChild(label);
    wrapper.appendChild(slider);
    wrapper.appendChild(valDisplay);
    return wrapper;
}

function renderLinks() {
    const svg = document.getElementById('network_svg');
    svg.innerHTML = '';
    const containerRect = svg.parentElement.getBoundingClientRect();

    function drawLine(fromEl, toEl, weight) {
        if (!fromEl || !toEl || weight === 0) return;
        const a = fromEl.getBoundingClientRect();
        const b = toEl.getBoundingClientRect();
        const x1 = a.left + a.width / 2 - containerRect.left;
        const y1 = a.top + a.height / 2 - containerRect.top;
        const x2 = b.left + b.width / 2 - containerRect.left;
        const y2 = b.top + b.height / 2 - containerRect.top;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1); line.setAttribute('y1', y1);
        line.setAttribute('x2', x2); line.setAttribute('y2', y2);
        line.setAttribute('stroke', weight >= 0 ? '#4CAF50' : '#e53935');
        line.setAttribute('stroke-width', Math.max(0.5, Math.abs(weight) * 4));
        line.setAttribute('stroke-linecap', 'round');
        svg.appendChild(line);
    }

    for (let i = 0; i < 2; i++)
        for (let j = 0; j < numHidden; j++)
            drawLine(document.getElementById(`input_${i}`), document.getElementById(`hidden_${j}`), weightsIn[i][j]);

    for (let j = 0; j < numHidden; j++)
        drawLine(document.getElementById(`hidden_${j}`), document.getElementById('output_0'), weightsOut[j]);
}

function updateNetwork() {
    let hiddenStates = new Array(numHidden).fill(0);

    for (let j = 0; j < numHidden; j++) {
        let sum = (inputs[0] * weightsIn[0][j]) + (inputs[1] * weightsIn[1][j]) + biasesHidden[j];
        hiddenStates[j] = sum > 0 ? 1 : 0;

        const neuronEl = document.getElementById(`hidden_${j}`);
        if (neuronEl) {
            neuronEl.textContent = hiddenStates[j];
            if (hiddenStates[j] === 1) neuronEl.classList.add('active');
            else neuronEl.classList.remove('active');
        }
    }

    let outputSum = biasOut;
    for (let j = 0; j < numHidden; j++)
        outputSum += hiddenStates[j] * weightsOut[j];

    const finalOutput = outputSum > 0 ? 1 : 0;
    const outputEl = document.getElementById('output_0');
    outputEl.textContent = finalOutput;
    if (finalOutput === 1) outputEl.classList.add('active');
    else outputEl.classList.remove('active');

    renderLinks();
}

initNetwork();
