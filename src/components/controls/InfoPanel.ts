import { TinyUNet } from "../../types/tiny";
import { countParameters } from "../../models/TinyModelGenerator";

export interface InfoPanelCallbacks {
  onNewInput: () => void;
  onNewWeights: () => void;
  onTimestepChange: (timestep: number) => void;
}

/**
 * Create the info panel for the Tiny U-Net visualization
 * Uses minimal styling with vanilla HTML elements (fieldsets, legends, etc.)
 */
export function createInfoPanel(
  model: TinyUNet,
  timestep: number,
  callbacks: InfoPanelCallbacks
): HTMLElement {
  // Create main panel with minimal positioning
  const panel = document.createElement('section');
  panel.style.position = 'absolute';
  panel.style.top = '10px';
  panel.style.left = '10px';
  panel.style.zIndex = '1000';
  panel.style.background = 'rgba(0, 0, 0, 0.8)';

  // Main fieldset wrapper
  const mainFieldset = document.createElement('fieldset');
  mainFieldset.style.borderRadius = '4px';
  const mainLegend = document.createElement('legend');

  // Collapsible toggle
  const toggleButton = document.createElement('span');
  toggleButton.textContent = '[-] ';
  toggleButton.style.cursor = 'pointer';
  toggleButton.style.userSelect = 'none';
  toggleButton.style.fontFamily = 'monospace';

  const titleSpan = document.createElement('span');
  titleSpan.textContent = 'Tiny Diffusion U-Net';

  mainLegend.appendChild(toggleButton);
  mainLegend.appendChild(titleSpan);
  mainLegend.style.cursor = 'pointer';

  // Collapsible content container
  const collapsibleContent = document.createElement('div');

  mainFieldset.appendChild(mainLegend);
  mainFieldset.appendChild(collapsibleContent);
  panel.appendChild(mainFieldset);

  // Collapse/expand handler
  let isCollapsed = false;
  mainLegend.addEventListener('click', () => {
    isCollapsed = !isCollapsed;
    toggleButton.textContent = isCollapsed ? '[+] ' : '[-] ';
    collapsibleContent.style.display = isCollapsed ? 'none' : 'block';
  });

  // Controls fieldset
  const controlsFieldset = document.createElement('fieldset');
  controlsFieldset.style.borderRadius = '4px';
  const controlsLegend = document.createElement('legend');
  controlsLegend.textContent = 'Controls';
  controlsFieldset.appendChild(controlsLegend);

  const newInputBtn = document.createElement('button');
  newInputBtn.textContent = '🔄 New Input';
  newInputBtn.addEventListener('click', callbacks.onNewInput);

  const newWeightsBtn = document.createElement('button');
  newWeightsBtn.textContent = '🎲 New Weights';
  newWeightsBtn.addEventListener('click', callbacks.onNewWeights);

  controlsFieldset.appendChild(document.createElement('br'));
  controlsFieldset.appendChild(newInputBtn);
  controlsFieldset.appendChild(document.createElement('br'));
  controlsFieldset.appendChild(document.createElement('br'));
  controlsFieldset.appendChild(newWeightsBtn);
  controlsFieldset.appendChild(document.createElement('br'));
  controlsFieldset.appendChild(document.createElement('br'));

  // Timestep slider
  const sliderContainer = document.createElement('div');
  const sliderId = 'slider-timestep';

  const sliderLabel = document.createElement('label');
  sliderLabel.htmlFor = sliderId;
  sliderLabel.textContent = `Timestep t: ${timestep.toFixed(2)}`;

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.id = sliderId;
  slider.min = '0';
  slider.max = '1';
  slider.step = '0.01';
  slider.value = timestep.toString();

  slider.addEventListener('input', () => {
    const newValue = parseFloat(slider.value);
    sliderLabel.textContent = `Timestep t: ${newValue.toFixed(2)}`;
    callbacks.onTimestepChange(newValue);
  });

  sliderContainer.appendChild(sliderLabel);
  sliderContainer.appendChild(document.createElement('br'));
  sliderContainer.appendChild(slider);
  controlsFieldset.appendChild(sliderContainer);

  collapsibleContent.appendChild(controlsFieldset);

  // Legend fieldset
  const legendFieldset = document.createElement('fieldset');
  legendFieldset.style.borderRadius = '4px';
  const legendLegendEl = document.createElement('legend');
  legendLegendEl.textContent = 'Legend';
  legendFieldset.appendChild(legendLegendEl);

  const legendInfo = document.createElement('div');
  legendInfo.innerHTML = `
    <ul>
      <li>🔵 <strong>Blue</strong> = Positive weights</li>
      <li>🔴 <strong>Red</strong> = Negative weights</li>
      <li>🧊 <strong>Size</strong> = Weight magnitude</li>
      <li>⬜ <small>Planes = data through the network</small></li>
    </ul>
  `;
  legendFieldset.appendChild(legendInfo);
  collapsibleContent.appendChild(legendFieldset);

  // Store slider reference for updates
  (panel as any)._timestepSlider = slider;
  (panel as any)._timestepLabel = sliderLabel;

  return panel;
}

/**
 * Update the timestep display in the info panel
 */
export function updateInfoPanelTimestep(panel: HTMLElement, timestep: number): void {
  const slider = (panel as any)._timestepSlider as HTMLInputElement;
  const label = (panel as any)._timestepLabel as HTMLLabelElement;
  if (slider && label) {
    slider.value = timestep.toString();
    label.textContent = `Timestep t: ${timestep.toFixed(2)}`;
  }
}
