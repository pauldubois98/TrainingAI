// State
let inputs = [0, 0];
let numHiddenLayers = 1;
let hiddenLayerSizes = [3]; // one entry per hidden layer

// weights[l][i][j]: weight from neuron i in layer l to neuron j in layer l+1
// biases[l][j]:     bias for neuron j in layer l+1
let weights = [];
let biases = [];

function getAllLayerSizes() {
    return [2, ...hiddenLayerSizes, 1];
}

// l is the absolute layer index (0 = input, last = output)
function hiddenLayerName(l, short = false) {
    if (numHiddenLayers === 1) return short ? 'H' : 'Hidden';
    return short ? `H${l}` : `Hidden ${l}`;
}

// Called when the hidden layer count input changes
function onNumLayersChange() {
    const el = document.getElementById('num_hidden_layers');
    numHiddenLayers = Math.max(0, Math.min(3, parseInt(el.value) || 0));
    el.value = numHiddenLayers;
    // Preserve existing sizes; pad with default 3
    hiddenLayerSizes = hiddenLayerSizes.slice(0, numHiddenLayers);
    while (hiddenLayerSizes.length < numHiddenLayers) hiddenLayerSizes.push(3);
    renderLayerSizeControls();
    initNetwork();
}

function renderLayerSizeControls() {
    const container = document.getElementById('layer_size_controls');
    container.innerHTML = '';
    for (let l = 0; l < numHiddenLayers; l++) {
        const wrapper = document.createElement('span');
        wrapper.className = 'layer-size-control';
        const label = document.createElement('label');
        label.textContent = `Layer ${l + 1} neurons:`;
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '1';
        input.max = '10';
        input.value = hiddenLayerSizes[l];
        const idx = l;
        input.onchange = () => {
            hiddenLayerSizes[idx] = Math.max(1, Math.min(10, parseInt(input.value) || 1));
            input.value = hiddenLayerSizes[idx];
            initNetwork();
        };
        wrapper.appendChild(label);
        wrapper.appendChild(input);
        container.appendChild(wrapper);
    }
}

function initNetwork() {
    const sizes = getAllLayerSizes();
    weights = [];
    biases = [];
    for (let l = 0; l < sizes.length - 1; l++) {
        weights[l] = Array.from({length: sizes[l]}, () => new Array(sizes[l + 1]).fill(0));
        biases[l] = new Array(sizes[l + 1]).fill(0);
    }
    renderNeurons();
    renderSliders();
    updateNetwork();
}

function toggleInput(index) {
    inputs[index] = inputs[index] === 0 ? 1 : 0;
    const btn = document.getElementById(`neuron_0_${index}`);
    btn.textContent = inputs[index];
    if (inputs[index] === 1) btn.classList.add('active');
    else btn.classList.remove('active');
    updateNetwork();
}

function renderNeurons() {
    const container = document.getElementById('network_layers');
    container.innerHTML = '';
    const sizes = getAllLayerSizes();

    for (let l = 0; l < sizes.length; l++) {
        const layerDiv = document.createElement('div');
        layerDiv.className = 'layer';

        const labelDiv = document.createElement('div');
        if (l === 0) labelDiv.textContent = 'Inputs';
        else if (l === sizes.length - 1) labelDiv.textContent = 'Output';
        else labelDiv.textContent = hiddenLayerName(l);
        layerDiv.appendChild(labelDiv);

        for (let j = 0; j < sizes[l]; j++) {
            const neuron = document.createElement('div');
            neuron.id = `neuron_${l}_${j}`;
            neuron.className = 'neuron';
            if (l === 0) {
                neuron.classList.add('clickable');
                const idx = j;
                neuron.onclick = () => toggleInput(idx);
                neuron.textContent = inputs[j];
                if (inputs[j] === 1) neuron.classList.add('active');
            } else {
                neuron.textContent = '0';
            }
            layerDiv.appendChild(neuron);
        }
        container.appendChild(layerDiv);
    }
}

function renderSliders() {
    const container = document.getElementById('sliders_content');
    container.innerHTML = '';
    const sizes = getAllLayerSizes();

    for (let l = 0; l < sizes.length - 1; l++) {
        const fromLabel = l === 0 ? 'In' : hiddenLayerName(l, true);
        const toLabel = l === sizes.length - 2 ? 'Out' : hiddenLayerName(l + 1, true);

        const wHeader = document.createElement('h3');
        wHeader.textContent = `${fromLabel} → ${toLabel} Weights`;
        container.appendChild(wHeader);

        for (let i = 0; i < sizes[l]; i++) {
            for (let j = 0; j < sizes[l + 1]; j++) {
                const fromName = l === 0 ? `In${i}` : `${hiddenLayerName(l, true)}_${i}`;
                const toName = l === sizes.length - 2 ? 'Out' : `${hiddenLayerName(l + 1, true)}_${j}`;
                const li = i, lj = j, ll = l;
                container.appendChild(createSliderMarkup(
                    `w_${l}_${i}_${j}`,
                    `W(${fromName}→${toName})`,
                    0.0,
                    (val) => { weights[ll][li][lj] = val; }
                ));
            }
        }

        const bHeader = document.createElement('h3');
        bHeader.textContent = `${toLabel} Biases`;
        container.appendChild(bHeader);

        for (let j = 0; j < sizes[l + 1]; j++) {
            const neuronName = l === sizes.length - 2 ? 'Out' : `${hiddenLayerName(l + 1, true)}_${j}`;
            const lj = j, ll = l;
            container.appendChild(createSliderMarkup(
                `b_${l}_${j}`,
                `B(${neuronName})`,
                0.0,
                (val) => { biases[ll][lj] = val; }
            ));
        }
    }
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
    const sizes = getAllLayerSizes();

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

    for (let l = 0; l < sizes.length - 1; l++) {
        for (let i = 0; i < sizes[l]; i++) {
            for (let j = 0; j < sizes[l + 1]; j++) {
                drawLine(
                    document.getElementById(`neuron_${l}_${i}`),
                    document.getElementById(`neuron_${l + 1}_${j}`),
                    weights[l][i][j]
                );
            }
        }
    }
}

function updateNetwork() {
    const sizes = getAllLayerSizes();
    let activations = [...inputs];

    for (let l = 0; l < sizes.length - 1; l++) {
        const next = [];
        for (let j = 0; j < sizes[l + 1]; j++) {
            let sum = biases[l][j];
            for (let i = 0; i < sizes[l]; i++) sum += activations[i] * weights[l][i][j];
            const out = sum > 0 ? 1 : 0;
            next.push(out);
            const el = document.getElementById(`neuron_${l + 1}_${j}`);
            if (el) {
                el.textContent = out;
                if (out === 1) el.classList.add('active');
                else el.classList.remove('active');
            }
        }
        activations = next;
    }

    renderLinks();
}

// Boot
renderLayerSizeControls();
initNetwork();
