import * as THREE from "three";

/**
 * Panel types in the application
 */
export enum PanelType {
  TRAINING_DATA = "trainingData",
  NEURAL_NETWORK = "neuralNetwork",
  PREDICTIONS = "predictions",
  POLYTOPES = "polytopes",
  ANALYTICAL_POLYTOPES = "analyticalPolytopes",
  LINES = "lines"
}

/**
 * Scene configuration - simplified for clarity
 */
export interface SceneConfig {
  dataGridSize: { width: number; height: number };
  camera: {
    fov: number;
    near: number;
    far: number;
    initialPosition: THREE.Vector3;
  };
  renderer: { backgroundColor: number; antialias: boolean };
  panelSpacing: number;
}

/**
 * Default configuration
 */
export const DEFAULT_SCENE_CONFIG: SceneConfig = {
  dataGridSize: {
    width: 10,
    height: 10
  },
  camera: {
    fov: 50,
    near: 0.1,
    far: 2000,
    initialPosition: new THREE.Vector3(0, 0, 5)
  },
  renderer: {
    backgroundColor: 0x1a1a1a,
    antialias: true
  },
  panelSpacing: 4
};
