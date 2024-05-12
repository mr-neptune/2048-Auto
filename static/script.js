document.addEventListener('keydown', function(event) {
    let action = null;
    switch (event.key) {
        case "ArrowUp":
            action = 0;
            break;
        case "ArrowRight":
            action = 1;
            break;
        case "ArrowDown":
            action = 2;
            break;
        case "ArrowLeft":
            action = 3;
            break;
        default:
            return; // Exit if it's not an arrow key
    }
    
    fetch('/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: action }),
    })
    .then(response => response.json())
    .then(data => {
        updateBoard(data.board, data.score, data.game_over);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

// Function to fetch and update the board with the initial game state
function fetchInitialState() {
    fetch('/init')
        .then(response => response.json())
        .then(data => {
            updateBoard(data.board, data.score, data.game_over);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

// Call fetchInitialState when the window loads
window.onload = function() {
    fetchInitialState();
};

function updateBoard(board, score, game_over) {
    const cells = document.querySelectorAll('.grid-cell');
    for (let i = 0; i < cells.length; i++) {
        const row = Math.floor(i / 4);
        const col = i % 4;
        const value = board[row][col];
        const cell = cells[i];

        // Clear previous classes
        cell.className = 'grid-cell';
        cell.textContent = value === 0 ? '' : value;

        // Add color based on the cell's value
        if (value === 2) cell.classList.add('tile-2');
        else if (value === 4) cell.classList.add('tile-4');
        else if (value === 8) cell.classList.add('tile-8');
        else if (value === 16) cell.classList.add('tile-16');
        else if (value === 32) cell.classList.add('tile-32');
        else if (value === 64) cell.classList.add('tile-64');
        else if (value > 64) cell.classList.add('tile-super');
    }
    updateScore(score)
    updateGameState(game_over)
}

function newGame() {
    fetch('/init', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        updateBoard(data.board, data.score, data.game_over);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function updateScore(score) {
    document.getElementById('current-score').textContent = score;
}


function updateGameState(game_over) {
    const gameContainer = document.getElementById('game-container');
    
    // Clear any existing overlay first
    const existingOverlay = document.querySelector('.game-over-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    if (game_over) {
        // Create the game over overlay
        const overlay = document.createElement('div');
        overlay.className = 'game-over-overlay';
        overlay.innerHTML = `
            <span>Game Over!</span>
            <button class="new-game-button" onclick="newGame()">Try again</button>
        `;

        // Append the overlay to the game container
        gameContainer.appendChild(overlay);
    }
}
    