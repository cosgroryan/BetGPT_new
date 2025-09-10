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
                    <td colspan="12" class="text-center text-muted py-4">
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
    }
};

// Global functions for HTML onclick handlers
window.getRecommendations = () => App.getRecommendations();
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
