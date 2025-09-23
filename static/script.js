const GRID_SIZE = 4;
const MOVE_ANIMATION_DURATION = 180; // milliseconds
const ANIMATION_BUFFER = 60;
const AUTO_PLAY_SPEEDS = {
    slow: 900,
    normal: 450,
    fast: 220,
};
const AUTO_PLAY_CHECK_INTERVAL = 60;
const NOT_IMPLEMENTED_METHODS = new Set(['algorithm', 'smart']);

let tileContainer = null;
let tilesState = new Map();
let tileIdCounter = 0;
let lastAction = null;
let isAnimating = false;
let cellPositions = [];
let isWaitingForResponse = false;
let autoPlayTimer = null;
let autoPlayActive = false;
let autoPlayButton = null;
let controlStatusEl = null;

const KNOWN_TILE_CLASSES = new Set([
    'tile-2', 'tile-4', 'tile-8', 'tile-16', 'tile-32', 'tile-64',
    'tile-128', 'tile-256', 'tile-512', 'tile-1024', 'tile-2048', 'tile-super'
]);

document.addEventListener('DOMContentLoaded', () => {
    tileContainer = document.querySelector('.tile-container');
    autoPlayButton = document.getElementById('auto-play-toggle');
    controlStatusEl = document.getElementById('control-status');
    cacheCellPositions();
    window.addEventListener('resize', () => {
        cacheCellPositions();
        refreshTilePositions();
    });
    setupAutoPlayControls();
    fetchInitialState();
});

document.addEventListener('keydown', (event) => {
    if (isAnimating || isWaitingForResponse) {
        return;
    }

    let action = null;
    switch (event.key) {
        case 'ArrowUp':
            action = 0;
            break;
        case 'ArrowRight':
            action = 1;
            break;
        case 'ArrowDown':
            action = 2;
            break;
        case 'ArrowLeft':
            action = 3;
            break;
        default:
            return;
    }

    const dispatched = sendMove(action);
    if (dispatched) {
        event.preventDefault();
    }
});

function sendMove(action) {
    if (action === null || action === undefined) {
        return false;
    }

    if (isAnimating || isWaitingForResponse) {
        return false;
    }

    lastAction = action;
    isWaitingForResponse = true;

    fetch('/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action }),
    })
        .then((response) => response.json())
        .then((data) => {
            isWaitingForResponse = false;
            updateBoard(data.board, data.score, data.game_over);
        })
        .catch((error) => {
            console.error('Error:', error);
            isWaitingForResponse = false;
            if (autoPlayActive) {
                stopAutoPlay('Error contacting the server. Auto play stopped.', 'warning');
            } else {
                setControlStatus('Error contacting the server.', 'warning');
            }
        });

    return true;
}

function fetchInitialState() {
    lastAction = null;
    isWaitingForResponse = true;
    fetch('/init')
        .then((response) => response.json())
        .then((data) => {
            isWaitingForResponse = false;
            updateBoard(data.board, data.score, data.game_over);
        })
        .catch((error) => {
            console.error('Error:', error);
            isWaitingForResponse = false;
            if (autoPlayActive) {
                stopAutoPlay('Error initializing the game. Auto play stopped.', 'warning');
            } else {
                setControlStatus('Error initializing the game.', 'warning');
            }
        });
}

function updateBoard(board, score, gameOver) {
    updateScore(score);

    const canAnimate = lastAction !== null && tilesState.size > 0;
    if (canAnimate) {
        const simulation = simulateMove(lastAction, board);
        if (simulation && (simulation.moved || simulation.spawnTiles.length > 0)) {
            tileIdCounter = simulation.nextId;
            playAnimation(simulation, board, gameOver);
        } else {
            renderBoardInstant(board);
            updateGameState(gameOver);
        }
    } else {
        renderBoardInstant(board);
        updateGameState(gameOver);
    }

    lastAction = null;
}

function renderBoardInstant(board) {
    if (!tileContainer) {
        return;
    }

    if (!cellPositions.length) {
        cacheCellPositions();
    }

    tileContainer.innerHTML = '';
    tilesState.clear();

    for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
            const value = board[row][col];
            if (value === 0) {
                continue;
            }
            const tileState = createTileState(value, row, col);
            tileContainer.appendChild(tileState.element);
            tilesState.set(tileState.id, tileState);
        }
    }

    isAnimating = false;
}

function playAnimation(simulation, finalBoard, gameOver) {
    const { moves, resultTiles, spawnTiles, removedTiles } = simulation;
    isAnimating = true;

    moves.forEach((move) => {
        const tileState = tilesState.get(move.id);
        if (!tileState) {
            return;
        }
        tileState.row = move.toRow;
        tileState.col = move.toCol;
        requestAnimationFrame(() => {
            setTilePosition(tileState.element, move.toRow, move.toCol);
        });
    });

    const cleanupDelay = MOVE_ANIMATION_DURATION + ANIMATION_BUFFER;

    setTimeout(() => {
        removedTiles.forEach((id) => {
            const tileState = tilesState.get(id);
            if (tileState) {
                tileState.element.remove();
                tilesState.delete(id);
            }
        });

        resultTiles.forEach((tile) => {
            if (tilesState.has(tile.id)) {
                const tileState = tilesState.get(tile.id);
                tileState.value = tile.value;
                tileState.row = tile.row;
                tileState.col = tile.col;
                tileState.element.textContent = tile.value;
                updateTileClasses(tileState.element, tile.value);
                setTilePosition(tileState.element, tile.row, tile.col);
            } else {
                const tileState = createTileState(tile.value, tile.row, tile.col, tile.id);
                tileState.element.classList.add('tile-merged');
                tileContainer.appendChild(tileState.element);
                tilesState.set(tile.id, tileState);
            }
        });

        spawnTiles.forEach((tile) => {
            const tileState = createTileState(tile.value, tile.row, tile.col, tile.id);
            tileState.element.classList.add('tile-new');
            tileContainer.appendChild(tileState.element);
            tilesState.set(tile.id, tileState);
        });

        ensureBoardIntegrity(finalBoard);
        updateGameState(gameOver);
        isAnimating = false;
    }, cleanupDelay);
}

function simulateMove(action, finalBoard) {
    if (action === null) {
        return null;
    }

    const board = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(null));
    tilesState.forEach((tile) => {
        board[tile.row][tile.col] = {
            id: tile.id,
            value: tile.value,
            row: tile.row,
            col: tile.col,
        };
    });

    let moved = false;
    let tempNextId = tileIdCounter;
    const moves = [];
    const resultTiles = [];
    const removedTiles = new Set();

    const processMerge = (first, second, targetRow, targetCol) => {
        const newId = tempNextId++;
        moves.push({ id: first.id, toRow: targetRow, toCol: targetCol, mergeInto: newId });
        moves.push({ id: second.id, toRow: targetRow, toCol: targetCol, mergeInto: newId });
        resultTiles.push({ id: newId, value: first.value * 2, row: targetRow, col: targetCol, isMergeResult: true });
        removedTiles.add(first.id);
        removedTiles.add(second.id);
        if (first.row !== targetRow || first.col !== targetCol) {
            moved = true;
        }
        if (second.row !== targetRow || second.col !== targetCol) {
            moved = true;
        }
        moved = true;
    };

    const processMove = (tile, targetRow, targetCol) => {
        moves.push({ id: tile.id, toRow: targetRow, toCol: targetCol, mergeInto: null });
        resultTiles.push({ id: tile.id, value: tile.value, row: targetRow, col: targetCol, isMergeResult: false });
        if (tile.row !== targetRow || tile.col !== targetCol) {
            moved = true;
        }
    };

    switch (action) {
        case 0: // Up
            for (let col = 0; col < GRID_SIZE; col++) {
                const line = [];
                for (let row = 0; row < GRID_SIZE; row++) {
                    const cell = board[row][col];
                    if (cell) {
                        line.push({ ...cell });
                    }
                }
                let targetRow = 0;
                while (line.length) {
                    const current = line.shift();
                    if (line.length && line[0].value === current.value) {
                        const next = line.shift();
                        processMerge(current, next, targetRow, col);
                    } else {
                        processMove(current, targetRow, col);
                    }
                    targetRow++;
                }
            }
            break;
        case 1: // Right
            for (let row = 0; row < GRID_SIZE; row++) {
                const line = [];
                for (let col = GRID_SIZE - 1; col >= 0; col--) {
                    const cell = board[row][col];
                    if (cell) {
                        line.push({ ...cell });
                    }
                }
                let targetCol = GRID_SIZE - 1;
                while (line.length) {
                    const current = line.shift();
                    if (line.length && line[0].value === current.value) {
                        const next = line.shift();
                        processMerge(current, next, row, targetCol);
                    } else {
                        processMove(current, row, targetCol);
                    }
                    targetCol--;
                }
            }
            break;
        case 2: // Down
            for (let col = 0; col < GRID_SIZE; col++) {
                const line = [];
                for (let row = GRID_SIZE - 1; row >= 0; row--) {
                    const cell = board[row][col];
                    if (cell) {
                        line.push({ ...cell });
                    }
                }
                let targetRow = GRID_SIZE - 1;
                while (line.length) {
                    const current = line.shift();
                    if (line.length && line[0].value === current.value) {
                        const next = line.shift();
                        processMerge(current, next, targetRow, col);
                    } else {
                        processMove(current, targetRow, col);
                    }
                    targetRow--;
                }
            }
            break;
        case 3: // Left
            for (let row = 0; row < GRID_SIZE; row++) {
                const line = [];
                for (let col = 0; col < GRID_SIZE; col++) {
                    const cell = board[row][col];
                    if (cell) {
                        line.push({ ...cell });
                    }
                }
                let targetCol = 0;
                while (line.length) {
                    const current = line.shift();
                    if (line.length && line[0].value === current.value) {
                        const next = line.shift();
                        processMerge(current, next, row, targetCol);
                    } else {
                        processMove(current, row, targetCol);
                    }
                    targetCol++;
                }
            }
            break;
        default:
            return null;
    }

    const resultBoard = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
    resultTiles.forEach((tile) => {
        resultBoard[tile.row][tile.col] = tile.value;
    });

    const spawnTiles = [];
    let mismatch = false;

    outer: for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
            const finalValue = finalBoard[row][col];
            const simulatedValue = resultBoard[row][col];
            if (finalValue !== simulatedValue) {
                if (simulatedValue === 0 && finalValue > 0) {
                    const newId = tempNextId++;
                    spawnTiles.push({ id: newId, value: finalValue, row, col });
                    resultBoard[row][col] = finalValue;
                } else {
                    mismatch = true;
                    break outer;
                }
            }
        }
    }

    if (mismatch) {
        return null;
    }

    if (!boardsEqual(resultBoard, finalBoard)) {
        return null;
    }

    return {
        moves,
        resultTiles,
        spawnTiles,
        removedTiles,
        moved: moved || spawnTiles.length > 0,
        nextId: tempNextId,
    };
}

function cacheCellPositions() {
    if (!tileContainer) {
        return;
    }

    const containerRect = tileContainer.getBoundingClientRect();
    const rows = Array.from(document.querySelectorAll('.grid-row'));
    cellPositions = rows.map((rowEl) => {
        const cells = Array.from(rowEl.querySelectorAll('.grid-cell'));
        return cells.map((cellEl) => {
            const rect = cellEl.getBoundingClientRect();
            return {
                x: rect.left - containerRect.left,
                y: rect.top - containerRect.top,
                size: rect.width,
            };
        });
    });

    if (cellPositions[0] && cellPositions[0][0]) {
        tileContainer.style.setProperty('--tile-size', `${cellPositions[0][0].size}px`);
    }
}

function refreshTilePositions() {
    tilesState.forEach((tile) => {
        setTilePosition(tile.element, tile.row, tile.col);
    });
}

function setTilePosition(element, row, col) {
    if (!cellPositions[row] || !cellPositions[row][col]) {
        cacheCellPositions();
    }

    const position = cellPositions[row][col];
    element.style.setProperty('--x', `${position.x}px`);
    element.style.setProperty('--y', `${position.y}px`);
}

function createTileState(value, row, col, id = null) {
    const tileId = id !== null ? id : getNextTileId();
    const element = buildTileElement(tileId, value, row, col);
    return { id: tileId, value, row, col, element };
}

function buildTileElement(tileId, value, row, col) {
    const element = document.createElement('div');
    element.classList.add('tile');
    element.dataset.id = tileId;
    element.textContent = value;
    updateTileClasses(element, value);
    setTilePosition(element, row, col);
    element.addEventListener('animationend', () => {
        element.classList.remove('tile-new');
        element.classList.remove('tile-merged');
    });
    return element;
}

function updateTileClasses(element, value) {
    KNOWN_TILE_CLASSES.forEach((cls) => {
        element.classList.remove(cls);
    });

    const className = KNOWN_TILE_CLASSES.has(`tile-${value}`) ? `tile-${value}` : 'tile-super';
    element.classList.add(className);
}

function getNextTileId() {
    const id = tileIdCounter;
    tileIdCounter += 1;
    return id;
}

function updateScore(score) {
    document.getElementById('current-score').textContent = score;
}

function updateGameState(gameOver) {
    const gameContainer = document.getElementById('game-container');
    const existingOverlay = document.querySelector('.game-over-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    if (gameOver) {
        const overlay = document.createElement('div');
        overlay.className = 'game-over-overlay';
        overlay.innerHTML = `
            <span>Game Over!</span>
            <button class="new-game-button" onclick="newGame()">Try again</button>
        `;
        gameContainer.appendChild(overlay);
        if (autoPlayActive) {
            stopAutoPlay('Game over! Auto play stopped.', 'info');
        }
    }
}

function ensureBoardIntegrity(targetBoard) {
    const currentBoard = getCurrentBoard();
    if (!boardsEqual(currentBoard, targetBoard)) {
        console.warn('Board mismatch detected, re-rendering.');
        renderBoardInstant(targetBoard);
    }
}

function getCurrentBoard() {
    const board = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
    tilesState.forEach((tile) => {
        board[tile.row][tile.col] = tile.value;
    });
    return board;
}

function boardsEqual(boardA, boardB) {
    for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
            if (boardA[row][col] !== boardB[row][col]) {
                return false;
            }
        }
    }
    return true;
}

function setupAutoPlayControls() {
    if (!autoPlayButton) {
        return;
    }

    autoPlayButton.addEventListener('click', () => {
        if (autoPlayActive) {
            stopAutoPlay('Auto play stopped.', 'info');
        } else {
            startAutoPlay();
        }
    });

    const methodInputs = document.querySelectorAll('input[name="play-method"]');
    methodInputs.forEach((input) => {
        input.addEventListener('change', () => {
            if (autoPlayActive) {
                return;
            }
            if (NOT_IMPLEMENTED_METHODS.has(input.value)) {
                setControlStatus('This method will be available soon. Choose Random Movements to auto play.', 'info');
            } else {
                setControlStatus('', '');
            }
        });
    });

    if (!autoPlayActive) {
        setControlStatus('Choose a method and speed, then press Start Auto Play.', 'info');
    }
}

function startAutoPlay() {
    const method = getSelectedMethod();
    if (NOT_IMPLEMENTED_METHODS.has(method)) {
        setControlStatus('This method will be available soon. Please choose Random Movements to start.', 'warning');
        return;
    }

    if (autoPlayTimer) {
        clearTimeout(autoPlayTimer);
    }

    autoPlayActive = true;
    if (autoPlayButton) {
        autoPlayButton.classList.add('is-active');
        autoPlayButton.textContent = 'Stop Auto Play';
    }
    setControlStatus('Auto play started.', 'success');
    runAutoPlayStep();
}

function stopAutoPlay(message = null, statusType = 'info') {
    if (autoPlayTimer) {
        clearTimeout(autoPlayTimer);
        autoPlayTimer = null;
    }

    autoPlayActive = false;

    if (autoPlayButton) {
        autoPlayButton.classList.remove('is-active');
        autoPlayButton.textContent = 'Start Auto Play';
    }

    if (message) {
        setControlStatus(message, statusType);
    }
}

function runAutoPlayStep() {
    if (!autoPlayActive) {
        return;
    }

    if (isAnimating || isWaitingForResponse) {
        autoPlayTimer = setTimeout(runAutoPlayStep, AUTO_PLAY_CHECK_INTERVAL);
        return;
    }

    const method = getSelectedMethod();
    const action = selectActionForMethod(method);

    if (action === null) {
        stopAutoPlay('Selected method is not available yet. Please choose Random Movements.', 'warning');
        return;
    }

    const moveSent = sendMove(action);
    if (!moveSent) {
        autoPlayTimer = setTimeout(runAutoPlayStep, AUTO_PLAY_CHECK_INTERVAL);
        return;
    }

    const delay = getSelectedSpeedDelay();
    autoPlayTimer = setTimeout(runAutoPlayStep, delay);
}

function getSelectedMethod() {
    const selected = document.querySelector('input[name="play-method"]:checked');
    return selected ? selected.value : 'random';
}

function getSelectedSpeedDelay() {
    const selected = document.querySelector('input[name="play-speed"]:checked');
    if (selected && Object.prototype.hasOwnProperty.call(AUTO_PLAY_SPEEDS, selected.value)) {
        return AUTO_PLAY_SPEEDS[selected.value];
    }
    return AUTO_PLAY_SPEEDS.normal;
}

function selectActionForMethod(method) {
    switch (method) {
        case 'random':
            return Math.floor(Math.random() * 4);
        case 'algorithm':
        case 'smart':
        default:
            return null;
    }
}

function setControlStatus(message, type = '') {
    if (!controlStatusEl) {
        return;
    }

    controlStatusEl.textContent = message;
    controlStatusEl.classList.remove('warning', 'success', 'info');
    if (type) {
        controlStatusEl.classList.add(type);
    }
}

function newGame() {
    if (isAnimating || isWaitingForResponse) {
        return;
    }

    lastAction = null;
    isAnimating = false;
    isWaitingForResponse = true;
    fetch('/init', {
        method: 'GET',
    })
        .then((response) => response.json())
        .then((data) => {
            isWaitingForResponse = false;
            updateBoard(data.board, data.score, data.game_over);
        })
        .catch((error) => {
            console.error('Error:', error);
            isWaitingForResponse = false;
        });
}
