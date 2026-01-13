# Diffusion World Model Visualiser

## TODO

- [ ] init


## Setup

   ```
   npm install
   npm start
   ```
 `http://localhost:3000`

## Usage

The application initializes a 3D scene for machine learning visualizations using THREE.js and WebGPU.

## Background
- https://arxiv.org/abs/2405.12399
- https://diamond-wm.github.io/


## Technical Details (for the agents 🤖)

### Architecture Overview

This project implements a 3D machine learning visualization and training system using TypeScript, THREE.js with WebGPU, and reactive programming with RxJS. The architecture follows a modular design with clear separation of concerns.

### Technology Stack

- **TypeScript**: Type-safe development with modern ES features
- **THREE.js**: 3D graphics with WebGPU renderer for high performance
- **RxJS**: Reactive programming for state management and data flow
- **Vite**: Fast development server and build tool

### Data Flow Architecture

```
User Interaction
↓
ControlManager
↓
AppController
↓
DataManager/TrainingManager
↓
Training Updates
↓
AppState
↓
VisualizationManager
↓
SceneManager
↓
THREE.js Scene Updates
↓
WebGPU Renderer
↓
Browser Display
```

### Key Design Patterns

1. **Observer Pattern**: RxJS observables for reactive state management
2. **Singleton Pattern**: AppState for centralized state
3. **Manager Pattern**: Separate managers for different concerns
4. **Factory Pattern**: Visualization creation functions
5. **Facade Pattern**: ControlManager simplifies UI interactions

### Core Principles

- Keep things simple - complexity is the enemy
- Less code is better - prefer concise, readable solutions
- Always use current best practices and latest language features
- Assume latest versions of dependencies and packages

### Implementation Guidelines

- Use modern TypeScript features (latest syntax, strict typing)
- Favor declarative code over imperative when possible
- Leverage RxJS for reactive, declarative data flow
- Prefer functional programming patterns where appropriate
- Use destructuring, arrow functions, and modern ES features
- Avoid unnecessary abstractions
- Prefer composition over inheritance
- Write self-documenting code with clear variable and function names
