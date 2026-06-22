// State
let inputValueSet = [0, 1];
let inputs = [0, 0];
let globalActivation = 'relu'; // 'relu' | 'tanh' | 'linear' | 'per_layer'
let layerActivations = []; // used only when globalActivation === 'per_layer'
const activationFunctions = {
    step:    x => x > 0 ? 1 : 0,
    relu:    x => Math.max(0, x),
    sigmoid: x => 1 / (1 + Math.exp(-x)),
    tanh:    x => Math.tanh(x),
    linear:  x => x,
};
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
    const prev = [...layerActivations];
    layerActivations = Array.from({length: sizes.length - 1}, (_, l) => prev[l] || 'step');
    weights = [];
    biases = [];
    for (let l = 0; l < sizes.length - 1; l++) {
        weights[l] = Array.from({length: sizes[l]}, () => new Array(sizes[l + 1]).fill(0));
        biases[l] = new Array(sizes[l + 1]).fill(0);
    }
    renderNeurons();
    if (globalActivation === 'per_layer') renderLayerActivationControls();
    renderSliders();
    updateNetwork();
}

function onInputSetChange() {
    const sel = document.getElementById('input_value_set');
    const customRow = document.getElementById('custom_input_controls');
    if (sel.value === 'binary') {
        inputValueSet = [0, 1];
        customRow.style.display = 'none';
    } else if (sel.value === 'ternary') {
        inputValueSet = [-1, 0, 1];
        customRow.style.display = 'none';
    } else {
        customRow.style.display = '';
        onCustomInputChange();
        return;
    }
    inputs = inputs.map(v => inputValueSet.includes(v) ? v : inputValueSet[0]);
    renderNeurons();
    updateNetwork();
}

function onCustomInputChange() {
    const raw = document.getElementById('custom_input_values').value;
    const parsed = raw.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    if (parsed.length < 2) return;
    inputValueSet = parsed;
    inputs = inputs.map(v => inputValueSet.includes(v) ? v : inputValueSet[0]);
    renderNeurons();
    updateNetwork();
}

function onGlobalActivationChange() {
    const prev = globalActivation;
    globalActivation = document.getElementById('global_activation').value;
    const layerControls = document.getElementById('layer_activation_controls');
    const globalPlot = document.getElementById('global_activation_plot');
    if (globalActivation === 'per_layer') {
        if (prev !== 'per_layer')
            layerActivations = layerActivations.map(() => prev);
        renderLayerActivationControls();
        layerControls.style.display = '';
        globalPlot.style.display = 'none';
    } else {
        layerControls.style.display = 'none';
        globalPlot.style.display = '';
        drawActivationPlot(globalPlot, globalActivation);
    }
    updateNetwork();
}

function renderLayerActivationControls() {
    const container = document.getElementById('layer_activation_controls');
    container.innerHTML = '';
    const sizes = getAllLayerSizes();
    const FN_NAMES = ['step', 'relu', 'sigmoid', 'tanh', 'linear'];

    for (let l = 0; l < sizes.length - 1; l++) {
        const isOutput = l === sizes.length - 2;
        const layerLabel = isOutput ? 'Output' : hiddenLayerName(l + 1);

        const wrapper = document.createElement('span');
        wrapper.className = 'layer-size-control';

        const label = document.createElement('label');
        label.textContent = `${layerLabel} activation:`;

        const select = document.createElement('select');
        FN_NAMES.forEach(fn => {
            const opt = document.createElement('option');
            opt.value = fn;
            opt.textContent = fn.charAt(0).toUpperCase() + fn.slice(1);
            opt.selected = layerActivations[l] === fn;
            select.appendChild(opt);
        });

        const canvasEl = document.createElement('canvas');
        canvasEl.width = 100;
        canvasEl.height = 64;
        canvasEl.className = 'activation-plot';

        const ll = l;
        select.onchange = () => {
            layerActivations[ll] = select.value;
            drawActivationPlot(canvasEl, select.value);
            updateNetwork();
        };

        wrapper.appendChild(label);
        wrapper.appendChild(select);
        wrapper.appendChild(canvasEl);
        container.appendChild(wrapper);

        drawActivationPlot(canvasEl, layerActivations[l]);
    }
}

function drawActivationPlot(canvas, fnName) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = 6;
    const plotW = W - 2 * pad, plotH = H - 2 * pad;

    const xMin = -3, xMax = 3, yMin = -1.5, yMax = 1.5;
    const toX = x => pad + (x - xMin) / (xMax - xMin) * plotW;
    const toY = y => pad + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    ctx.clearRect(0, 0, W, H);

    // Axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad, toY(0)); ctx.lineTo(pad + plotW, toY(0)); // x-axis
    ctx.moveTo(toX(0), pad); ctx.lineTo(toX(0), pad + plotH); // y-axis
    ctx.stroke();

    // Curve
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5;
    const fn = activationFunctions[fnName];

    if (fnName === 'step') {
        const x0 = toX(0), y0 = toY(0), y1 = toY(1);
        ctx.beginPath(); ctx.moveTo(pad, y0); ctx.lineTo(x0, y0); ctx.stroke();
        ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(pad + plotW, y1); ctx.stroke();
    } else {
        ctx.save();
        ctx.beginPath(); ctx.rect(pad, pad, plotW, plotH); ctx.clip();
        ctx.beginPath();
        for (let px = 0; px <= plotW; px++) {
            const x = xMin + (px / plotW) * (xMax - xMin);
            const cy = toY(fn(x));
            px === 0 ? ctx.moveTo(pad + px, cy) : ctx.lineTo(pad + px, cy);
        }
        ctx.stroke();
        ctx.restore();
    }
}

function formatVal(v) {
    return parseFloat(v.toFixed(2)).toString();
}

const GREY  = [221, 221, 221];
const GREEN = [76, 175, 80];
const RED   = [229, 57, 53];

function setNeuronActivation(el, val) {
    el.textContent = formatVal(val);
    const t = Math.min(1, Math.abs(val));
    const target = val >= 0 ? GREEN : RED;
    const r = Math.round(GREY[0] + t * (target[0] - GREY[0]));
    const g = Math.round(GREY[1] + t * (target[1] - GREY[1]));
    const b = Math.round(GREY[2] + t * (target[2] - GREY[2]));
    el.style.backgroundColor = `rgb(${r},${g},${b})`;
    el.style.color = t > 0.4 ? 'white' : '#333';
    el.classList.remove('active', 'negative');
}

function toggleInput(index) {
    const currentIdx = inputValueSet.indexOf(inputs[index]);
    inputs[index] = inputValueSet[(currentIdx + 1) % inputValueSet.length];
    const el = document.getElementById(`neuron_0_${index}`);
    el.textContent = inputs[index];
    setNeuronActivation(el, inputs[index]);
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
                setNeuronActivation(neuron, inputs[j]);
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
            const fnName = globalActivation === 'per_layer' ? layerActivations[l] : globalActivation;
            const out = activationFunctions[fnName](sum);
            next.push(out);
            const el = document.getElementById(`neuron_${l + 1}_${j}`);
            if (el) {
                setNeuronActivation(el, out);

                const bias = biases[l][j];
                if (bias === 0) {
                    el.style.outline = 'none';
                } else {
                    const color = bias > 0 ? '#4CAF50' : '#e53935';
                    const width = 2 + Math.abs(bias) * 6;
                    el.style.outline = `${width}px solid ${color}`;
                }
            }
        }
        activations = next;
    }

    renderLinks();
}

// Boot
renderLayerSizeControls();
initNetwork();
drawActivationPlot(document.getElementById('global_activation_plot'), globalActivation);
