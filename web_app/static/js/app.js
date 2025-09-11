/**
 * BetGPT Web App JavaScript
 * Main application logic and utilities
 */

// Global application state
const AppState = {
    currentRace: null,
    currentPredictions: null,
    currentRecommendations: null,
    isLoading: false
};

// Utility functions
const Utils = {
    /**
     * Format a number as currency
     */
    formatCurrency: (amount) => {
        return new Intl.NumberFormat('en-AU', {
            style: 'currency',
            currency: 'AUD'
        }).format(amount);
    },

    /**
     * Format a percentage
     */
    formatPercentage: (value, decimals = 1) => {
        if (value === null || value === undefined) return '-';
        return `${parseFloat(value).toFixed(decimals)}%`;
    },

    /**
     * Format odds as dollar amount
     */
    formatOdds: (odds) => {
        if (odds === null || odds === undefined) return '-';
        return '$' + parseFloat(odds).toFixed(2);
    },

    /**
     * Calculate implied probability from odds
     */
    impliedProbability: (odds) => {
        if (!odds || odds <= 1) return null;
        return (1 / odds) * 100;
    },

    /**
     * Calculate edge percentage
     */
    calculateEdge: (modelProb, impliedProb) => {
        if (!modelProb || !impliedProb) return null;
        return modelProb - impliedProb;
    },

    /**
     * Show loading state
     */
    showLoading: () => {
        AppState.isLoading = true;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
        
        // Auto-hide loading modal after 30 seconds to prevent stuck state
        setTimeout(() => {
            if (AppState.isLoading) {
                console.warn('Loading modal auto-hidden due to timeout');
                Utils.hideLoading();
            }
        }, 30000);
    },

    /**
     * Hide loading state
     */
    hideLoading: () => {
        AppState.isLoading = false;
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
        // Force remove any backdrop that might be stuck
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => backdrop.remove());
        // Remove modal-open class from body
        document.body.classList.remove('modal-open');
        // Reset body padding if it was modified
        document.body.style.paddingRight = '';
    },

    /**
     * Show error message
     */
    showError: (message) => {
        // You could implement a toast notification system here
        alert('Error: ' + message);
    },

    /**
     * Show success message
     */
    showSuccess: (message) => {
        // You could implement a toast notification system here
        console.log('Success: ' + message);
    }
};

// API service
const API = {
    /**
     * Fetch race data
     */
    fetchRace: async (date, meetNo, raceNo) => {
        const response = await fetch(`/api/race/${date}/${meetNo}/${raceNo}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    },

    /**
     * Fetch predictions
     */
    fetchPredictions: async (date, meetNo, raceNo) => {
        const response = await fetch(`/api/predictions/${date}/${meetNo}/${raceNo}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    },

    /**
     * Get recommendations
     */
    getRecommendations: async (settings) => {
        const response = await fetch('/api/recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings)
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
};

// UI components
const UI = {
    /**
     * Update race header
     */
    updateRaceHeader: (raceData) => {
        const raceInfo = raceData.race_info;
        document.getElementById('raceTitle').textContent = 
            `${raceInfo.venue} | Race ${raceInfo.race_no || 'Unknown'}`;
        document.getElementById('raceDetails').textContent = 
            `Distance: ${raceInfo.distance} | Track: ${raceInfo.track_condition} | Weather: ${raceInfo.weather}`;
        document.getElementById('raceHeader').style.display = 'block';
    },

    /**
     * Update quick stats
     */
    updateQuickStats: (raceData) => {
        document.getElementById('fieldSize').textContent = raceData.field_size;
        
        // Find top pick
        let topPick = null;
        let topPickProb = 0;
        
        if (raceData.runners) {
            raceData.runners.forEach(runner => {
                const modelPred = runner.model_prediction || {};
                const winProb = modelPred.win_percentage || 0;
                
                if (winProb > topPickProb) {
                    topPick = runner.number;
                    topPickProb = winProb;
                }
            });
        }
        
        document.getElementById('topPick').textContent = topPick || '-';
    },

    /**
     * Get gradient class based on percentage value
     */
    getGradientClass: (percentage) => {
        if (percentage >= 20) return 'gradient-green-3';
        if (percentage >= 10) return 'gradient-green-2';
        if (percentage >= 5) return 'gradient-green-1';
        return '';
    },

    /**
     * Render runners table
     */
    renderRunnersTable: (raceData) => {
        const tbody = document.getElementById('runnersTableBody');
        tbody.innerHTML = '';
        
        if (!raceData.runners || raceData.runners.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="16" class="text-center text-muted py-4">
                        <i class="fas fa-horse fa-2x mb-3 d-block"></i>
                        No runners found
                    </td>
                </tr>
            `;
            return;
        }
        
        raceData.runners.forEach(runner => {
            const modelPred = runner.model_prediction || {};
            const winProb = modelPred.win_percentage || 0;
            const impliedProb = runner.implied_win_prob ? runner.implied_win_prob * 100 : 0;
            const impliedFormatted = impliedProb > 0 ? Utils.formatPercentage(impliedProb, 2) : '-';
            const edge = Utils.calculateEdge(winProb, impliedProb);
            const edgeFormatted = edge !== null ? Utils.formatPercentage(edge, 2) : '-';
            const edgeClass = edge > 0 ? 'edge-positive' : 'edge-negative';
            
            // Get gradient classes for percentages
            const modelGradientClass = UI.getGradientClass(winProb);
            const impliedGradientClass = UI.getGradientClass(impliedProb);
            
            const row = `
                <tr class="fade-in">
                    <td><strong>${runner.number}</strong></td>
                    <td>
                        ${runner.name}
                        ${modelPred.new_horse ? '<span class="badge bg-warning ms-1">NEW</span>' : ''}
                    </td>
                    <td>${runner.jockey}</td>
                    <td>${runner.barrier}</td>
                    <td>${runner.weight || '-'}</td>
                    <td class="form-cell" title="${runner.form || 'No form data'}">${runner.form ? (runner.form.length > 10 ? runner.form.substring(0, 10) + '...' : runner.form) : '-'}</td>
                    <td class="speedmap-cell">${runner.speedmap || '-'}</td>
                    <td class="edge-cell">${runner.edge || '-'}</td>
                    <td>${Utils.formatOdds(runner.odds.win_fixed)}</td>
                    <td>${Utils.formatOdds(runner.odds.win_tote)}</td>
                    <td>${Utils.formatOdds(runner.odds.place_fixed)}</td>
                    <td>${Utils.formatOdds(runner.odds.place_tote)}</td>
                    <td class="${modelGradientClass}">
                        <strong>${Utils.formatPercentage(winProb, 2)}</strong>
                    </td>
                    <td class="${impliedGradientClass}">
                        <strong>${impliedFormatted}</strong>
                    </td>
                    <td class="${edgeClass}">
                        <strong>${edgeFormatted}</strong>
                    </td>
                    <td>${Utils.formatOdds(modelPred.fair_odds)}</td>
                </tr>
            `;
            tbody.innerHTML += row;
        });
    },

    /**
     * Render recommendations table
     */
    renderRecommendationsTable: (recommendations) => {
        const tbody = document.getElementById('recommendationsTableBody');
        tbody.innerHTML = '';
        
        if (!recommendations.recommendations || recommendations.recommendations.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="10" class="text-center text-muted py-4">
                        <i class="fas fa-lightbulb fa-2x mb-3 d-block"></i>
                        No recommendations found
                    </td>
                </tr>
            `;
            return;
        }
        
        recommendations.recommendations.forEach(rec => {
            // Get gradient classes for percentages
            const marketGradientClass = UI.getGradientClass(rec.market_pct);
            const modelGradientClass = UI.getGradientClass(rec.model_pct);
            const blendGradientClass = UI.getGradientClass(rec.blend_pct);
            
            const row = `
                <tr class="fade-in">
                    <td><strong>#${rec.number} ${rec.runner}</strong></td>
                    <td>${rec.market_label}</td>
                    <td>${Utils.formatOdds(rec.odds)}</td>
                    <td class="${marketGradientClass}">
                        <strong>${Utils.formatPercentage(rec.market_pct, 2)}</strong>
                    </td>
                    <td class="${modelGradientClass}">
                        <strong>${Utils.formatPercentage(rec.model_pct, 2)}</strong>
                    </td>
                    <td class="${blendGradientClass}">
                        <strong>${Utils.formatPercentage(rec.blend_pct, 2)}</strong>
                    </td>
                    <td class="edge-positive">
                        <strong>${Utils.formatPercentage(rec.edge_pct, 2)}</strong>
                    </td>
                    <td>${Utils.formatOdds(rec.ev)}</td>
                    <td>${Utils.formatPercentage(rec.kelly_pct, 2)}</td>
                    <td class="text-primary">
                        <strong>${Utils.formatCurrency(rec.stake)}</strong>
                    </td>
                </tr>
            `;
            tbody.innerHTML += row;
        });
    },

    /**
     * Show recommendations panel
     */
    showRecommendationsPanel: () => {
        document.getElementById('recommendationsPanel').style.display = 'block';
        document.getElementById('recommendationsPanel').scrollIntoView({ 
            behavior: 'smooth' 
        });
    }
};

// Main application logic
const App = {
    /**
     * Initialize the application
     */
    init: () => {
        // Set today's date as default
        document.getElementById('raceDate').value = new Date().toISOString().split('T')[0];
        
        // Bind event listeners
        App.bindEventListeners();
        
        console.log('BetGPT Web App initialized');
    },

    /**
     * Bind event listeners
     */
    bindEventListeners: () => {
        // Race form submission
        document.getElementById('raceForm').addEventListener('submit', (e) => {
            e.preventDefault();
            App.loadRace();
        });

        // Recommendation settings change
        ['betType', 'market', 'modelWeight', 'minEdge', 'bankroll', 'kellyFraction'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                if (AppState.currentRace) {
                    App.getRecommendations();
                }
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // ESC key to clear stuck modals
            if (e.key === 'Escape') {
                if (AppState.isLoading) {
                    console.log('ESC pressed - clearing loading modal');
                    Utils.hideLoading();
                }
            }
            // Ctrl+Shift+D for debug panel
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                App.showDebugPanel();
            }
        });
    },

    /**
     * Load race data
     */
    loadRace: async () => {
        const date = document.getElementById('raceDate').value;
        const meetNo = document.getElementById('meetNumber').value;
        const raceNo = document.getElementById('raceNumber').value;
        
        if (!date || !meetNo || !raceNo) {
            Utils.showError('Please fill in all fields');
            return;
        }
        
        try {
            Utils.showLoading();
            
            // Clear any existing results and reset UI state
            App.clearResults();
            
            // Add timeout to prevent hanging
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
            
            const raceData = await API.fetchRace(date, meetNo, raceNo);
            clearTimeout(timeoutId);
            
            if (raceData.error) {
                throw new Error(raceData.error);
            }
            
            AppState.currentRace = raceData;
            
            // Update UI
            UI.updateRaceHeader(raceData);
            UI.updateQuickStats(raceData);
            UI.renderRunnersTable(raceData);
            
            Utils.showSuccess('Race data loaded successfully');
            
        } catch (error) {
            console.error('Error loading race:', error);
            if (error.name === 'AbortError') {
                Utils.showError('Request timed out. Please try again.');
            } else {
                Utils.showError(error.message || 'Failed to load race data');
            }
        } finally {
            Utils.hideLoading();
        }
    },

    /**
     * Get betting recommendations
     */
    getRecommendations: async () => {
        if (!AppState.currentRace) {
            Utils.showError('Please load a race first');
            return;
        }
        
        const settings = {
            date: document.getElementById('raceDate').value,
            meet_no: parseInt(document.getElementById('meetNumber').value),
            race_no: parseInt(document.getElementById('raceNumber').value),
            bet_type: document.getElementById('betType').value,
            market: document.getElementById('market').value,
            model_weight: parseFloat(document.getElementById('modelWeight').value),
            min_edge: parseFloat(document.getElementById('minEdge').value),
            bankroll: parseFloat(document.getElementById('bankroll').value),
            kelly_fraction: parseFloat(document.getElementById('kellyFraction').value)
        };
        
        try {
            Utils.showLoading();
            
            const recommendations = await API.getRecommendations(settings);
            
            if (recommendations.error) {
                throw new Error(recommendations.error);
            }
            
            AppState.currentRecommendations = recommendations;
            
            // Update UI
            UI.renderRecommendationsTable(recommendations);
            UI.showRecommendationsPanel();
            
            Utils.showSuccess('Recommendations generated successfully');
            
        } catch (error) {
            console.error('Error getting recommendations:', error);
            Utils.showError(error.message);
        } finally {
            Utils.hideLoading();
        }
    },

    /**
     * Show debug panel
     */
    showDebugPanel: () => {
        const debugInfo = {
            currentRace: AppState.currentRace ? 'Loaded' : 'None',
            isLoading: AppState.isLoading,
            modalBackdrops: document.querySelectorAll('.modal-backdrop').length,
            bodyClasses: document.body.className,
            activeModals: document.querySelectorAll('.modal.show').length
        };
        
        console.log('Debug Info:', debugInfo);
        alert(`Debug Info:
Current Race: ${debugInfo.currentRace}
Loading: ${debugInfo.isLoading}
Modal Backdrops: ${debugInfo.modalBackdrops}
Body Classes: ${debugInfo.bodyClasses}
Active Modals: ${debugInfo.activeModals}

Press OK to clear stuck modals.`);
        
        // Auto-clear if there are stuck modals
        if (debugInfo.modalBackdrops > 0 || debugInfo.activeModals > 0) {
            window.clearModals();
        }
    },

    /**
     * Get race results
     */
    getRaceResults: async () => {
        if (!AppState.currentRace) {
            Utils.showError('Please load a race first');
            return;
        }
        
        const raceInfo = AppState.currentRace.race_info;
        if (!raceInfo) {
            Utils.showError('Race information not available');
            return;
        }
        
        const { date, meet_no, race_no } = raceInfo;
        
        console.log('Getting results for:', { date, meet_no, race_no });
        
        try {
            Utils.showLoading();
            
            const response = await fetch(`/api/results/${date}/${meet_no}/${race_no}`);
            const data = await response.json();
            
            if (data.success && data.race_completed) {
                App.displayRaceResults(data.results);
                // Show results button and hide it after clicking
                document.getElementById('resultsBtn').style.display = 'none';
            } else {
                Utils.showError(data.message || 'Results not available for this race');
            }
            
        } catch (error) {
            console.error('Error getting race results:', error);
            Utils.showError(`Failed to get race results: ${error.message}`);
        } finally {
            Utils.hideLoading();
        }
    },

    /**
     * Display race results with highlighting
     */
    displayRaceResults: (resultsData) => {
        const { race_info, results } = resultsData;
        
        // Add results mode class to disable green gradients
        document.body.classList.add('results-mode');
        
        // Update race header to show results
        const raceTitle = document.getElementById('raceTitle');
        raceTitle.innerHTML = `${race_info.name} <span class="badge bg-success ms-2">COMPLETED</span>`;
        
        // Create results table
        const resultsTable = document.createElement('div');
        resultsTable.className = 'card shadow-sm results-table';
        resultsTable.innerHTML = `
            <div class="card-header bg-success text-white">
                <h5 class="card-title mb-0">
                    <i class="fas fa-trophy me-2"></i>Race Results
                </h5>
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-hover mb-0">
                         <thead>
                             <tr>
                                 <th>Position</th>
                                 <th>Number</th>
                                 <th>Horse</th>
                                 <th>Jockey</th>
                                 <th>Margin</th>
                                 <th>Distance</th>
                             </tr>
                         </thead>
                        <tbody id="resultsTableBody">
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        // Insert results table after race header
        const raceHeader = document.getElementById('raceHeader');
        raceHeader.insertAdjacentElement('afterend', resultsTable);
        
        // Populate results table
        const tbody = document.getElementById('resultsTableBody');
        tbody.innerHTML = '';
        
        results.forEach(result => {
            const row = document.createElement('tr');
            
            // Determine position class
            let positionClass = 'result-position-other';
            let badgeClass = 'pos-other';
            if (result.position === 1) {
                positionClass = 'result-position-1';
                badgeClass = 'pos-1';
            } else if (result.position === 2) {
                positionClass = 'result-position-2';
                badgeClass = 'pos-2';
            } else if (result.position === 3) {
                positionClass = 'result-position-3';
                badgeClass = 'pos-3';
            }
            
             row.className = positionClass;
             
             row.innerHTML = `
                 <td>
                     <span class="position-badge ${badgeClass}">${result.position}</span>
                 </td>
                 <td><strong>${result.number}</strong></td>
                 <td><strong>${result.name}</strong></td>
                 <td>${result.jockey}</td>
                 <td>${result.margin || '-'}</td>
                 <td>${result.distance || '-'}</td>
             `;
            
            tbody.appendChild(row);
        });
        
        // Update runners table to highlight results
        App.highlightRunnersWithResults(results);
        
        // Scroll to results
        resultsTable.scrollIntoView({ behavior: 'smooth' });
    },

    /**
     * Highlight runners in the main table with their results
     */
    highlightRunnersWithResults: (results) => {
        const runnersTable = document.getElementById('runnersTableBody');
        const rows = runnersTable.querySelectorAll('tr');
        
        // Create a map of runner numbers to results
        const resultsMap = {};
        results.forEach(result => {
            resultsMap[result.number] = result;
        });
        
        rows.forEach(row => {
            const numberCell = row.querySelector('td:first-child');
            if (numberCell) {
                const runnerNumber = numberCell.textContent.trim();
                const result = resultsMap[runnerNumber];
                
                if (result) {
                    // Add result highlighting
                    let positionClass = 'result-position-other';
                    if (result.position === 1) {
                        positionClass = 'result-position-1';
                    } else if (result.position === 2) {
                        positionClass = 'result-position-2';
                    } else if (result.position === 3) {
                        positionClass = 'result-position-3';
                    }
                    
                    row.classList.add(positionClass);
                    
                    // Add position indicator to the name
                    const nameCell = row.querySelector('td:nth-child(2)');
                    if (nameCell) {
                        const positionBadge = document.createElement('span');
                        positionBadge.className = `badge bg-secondary ms-2`;
                        positionBadge.textContent = `${result.position}${App.getOrdinalSuffix(result.position)}`;
                        nameCell.appendChild(positionBadge);
                    }
                }
            }
        });
    },

    /**
     * Get ordinal suffix for position numbers
     */
    getOrdinalSuffix: (num) => {
        const j = num % 10;
        const k = num % 100;
        if (j === 1 && k !== 11) {
            return "st";
        }
        if (j === 2 && k !== 12) {
            return "nd";
        }
        if (j === 3 && k !== 13) {
            return "rd";
        }
        return "th";
    },

    /**
     * Clear results and reset UI state
     */
    clearResults: () => {
        // Remove results mode class
        document.body.classList.remove('results-mode');
        
        // Remove any existing results table
        const existingResultsTable = document.querySelector('.results-table');
        if (existingResultsTable) {
            existingResultsTable.remove();
        }
        
        // Reset race header (remove COMPLETED badge)
        const raceTitle = document.getElementById('raceTitle');
        if (raceTitle) {
            raceTitle.innerHTML = raceTitle.textContent.replace(' COMPLETED', '');
        }
        
        // Clear results highlighting from runners table
        const runnersTable = document.getElementById('runnersTableBody');
        if (runnersTable) {
            const rows = runnersTable.querySelectorAll('tr');
            rows.forEach(row => {
                // Remove result position classes
                row.classList.remove('result-position-1', 'result-position-2', 'result-position-3', 'result-position-other');
                
                // Remove position badges from name cells
                const nameCell = row.querySelector('td:nth-child(2)');
                if (nameCell) {
                    const badges = nameCell.querySelectorAll('.badge.bg-secondary');
                    badges.forEach(badge => badge.remove());
                }
            });
        }
        
        // Show results button again
        const resultsBtn = document.getElementById('resultsBtn');
        if (resultsBtn) {
            resultsBtn.style.display = 'inline-block';
        }
    }
};

// Global functions for HTML onclick handlers
window.getRecommendations = () => App.getRecommendations();
window.getRaceResults = () => App.getRaceResults();
window.showAbout = () => {
    const modal = new bootstrap.Modal(document.getElementById('aboutModal'));
    modal.show();
};

// Emergency function to clear stuck modals
window.clearModals = () => {
    console.log('Clearing stuck modals...');
    Utils.hideLoading();
    // Remove all modal backdrops
    const backdrops = document.querySelectorAll('.modal-backdrop');
    backdrops.forEach(backdrop => backdrop.remove());
    // Remove modal-open class
    document.body.classList.remove('modal-open');
    // Reset body styles
    document.body.style.paddingRight = '';
    document.body.style.overflow = '';
    console.log('Modals cleared');
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', App.init);
