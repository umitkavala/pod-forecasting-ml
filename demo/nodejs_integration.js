/**
 * Pod Forecasting API - Node.js Integration Demo
 * 
 * Demonstrates how to integrate with the Pod Forecasting API using Node.js.
 */

const axios = require('axios');

const dotenv = require('dotenv');
dotenv.config();

// Configuration
const API_URL = process.env.API_URL || 'http://localhost:5000';
const API_KEY = process.env.API_KEY_NODE_SERVICE;
console.info(`Using API KEY: ${API_KEY}`);

// Check API key
if (!API_KEY) {
    console.error('Error: API_KEY_NODE_SERVICE environment variable not set');
    console.error('\nSet it with:');
    console.error('  export API_KEY_NODE_SERVICE="your-api-key-here"');
    process.exit(1);
}

// Headers
const headers = {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
};

/**
 * Test health check endpoint (no auth required)
 */
async function testHealthCheck() {
    console.log('\n' + '='.repeat(60));
    console.log('Testing health check...');
    console.log('='.repeat(60));
    
    try {
        const response = await axios.get(`${API_URL}/health`);
        
        if (response.status === 200) {
            const data = response.data;
            console.log('Health check passed');
            console.log(`  Status: ${data.status}`);
            console.log(`  Model loaded: ${data.model_loaded}`);
            return true;
        }
        
    } catch (error) {
        if (error.code === 'ECONNREFUSED') {
            console.log('Connection failed');
            console.log('  Make sure API is running: uvicorn api.main:app --port 5000');
        } else {
            console.log(`Error: ${error.message}`);
        }
        return false;
    }
}

/**
 * Test single prediction endpoint
 */
async function testSinglePrediction() {
    console.log('\n' + '='.repeat(60));
    console.log('Testing single prediction...');
    console.log('='.repeat(60));
    
    // Prediction data
    const data = {
        date: '2024-07-15',
        gmv: 9500000,
        users: 85000,
        marketing_cost: 175000
    };
    
    console.log('\nRequest:');
    console.log(`  Date: ${data.date}`);
    console.log(`  GMV: $${data.gmv.toLocaleString()}`);
    console.log(`  Users: ${data.users.toLocaleString()}`);
    console.log(`  Marketing Cost: $${data.marketing_cost.toLocaleString()}`);
    
    try {
        const response = await axios.post(
            `${API_URL}/predict`,
            data,
            { headers }
        );
        
        if (response.status === 200) {
            const result = response.data;
            const predictions = result.predictions;
            const confidence = result.confidence_intervals;
            
            console.log('\nPrediction successful!');
            console.log(`  Frontend pods: ${predictions.frontend_pods}`);
            console.log(`  Backend pods: ${predictions.backend_pods}`);
            console.log('  Confidence:');
            console.log(`    Frontend: [${confidence.frontend_pods.join(', ')}]`);
            console.log(`    Backend: [${confidence.backend_pods.join(', ')}]`);
            
            return true;
        }
        
    } catch (error) {
        if (error.response) {
            if (error.response.status === 401) {
                console.log('Authentication failed');
                console.log('  Check your API key');
            } else if (error.response.status === 503) {
                console.log('Service unavailable');
                console.log('  Model not loaded - check API logs');
            } else {
                console.log(`Request failed: ${error.response.status}`);
                console.log(`  Response: ${JSON.stringify(error.response.data)}`);
            }
        } else {
            console.log(`Error: ${error.message}`);
        }
        return false;
    }
}

/**
 * Test batch prediction endpoint
 */
async function testBatchPrediction() {
    console.log('\n' + '='.repeat(60));
    console.log('Testing batch predictions...');
    console.log('='.repeat(60));
    
    // Generate 3 days of predictions
    const baseDate = new Date('2024-07-15');
    
    const predictions = [];
    for (let i = 0; i < 3; i++) {
        const date = new Date(baseDate);
        date.setDate(date.getDate() + i);
        
        predictions.push({
            date: date.toISOString().split('T')[0],
            gmv: 9500000 + (i * 500000),
            users: 85000 + (i * 5000),
            marketing_cost: 175000 + (i * 5000)
        });
    }
    
    const data = { predictions };
    
    console.log(`\nRequest: ${predictions.length} predictions`);
    predictions.forEach(pred => {
        console.log(`  ${pred.date}: GMV=$${pred.gmv.toLocaleString()}, Users=${pred.users.toLocaleString()}`);
    });
    
    try {
        const response = await axios.post(
            `${API_URL}/forecast/batch`,
            data,
            { headers }
        );
        
        if (response.status === 200) {
            const result = response.data;
            const count = result.count;
            const results = result.predictions;
            
            console.log('\nBatch prediction successful!');
            console.log(`  Processed: ${count} predictions`);
            console.log('\n  Results:');
            
            results.forEach(pred => {
                const date = pred.date;
                const fe = pred.predictions.frontend_pods;
                const be = pred.predictions.backend_pods;
                console.log(`    ${date}: FE=${fe}, BE=${be}`);
            });
            
            return true;
        }
        
    } catch (error) {
        if (error.response) {
            if (error.response.status === 401) {
                console.log('Authentication failed');
            } else {
                console.log(`Request failed: ${error.response.status}`);
                console.log(`  Response: ${JSON.stringify(error.response.data)}`);
            }
        } else {
            console.log(`Error: ${error.message}`);
        }
        return false;
    }
}

/**
 * Test authentication with invalid key
 */
async function testInvalidAuth() {
    console.log('\n' + '='.repeat(60));
    console.log('Testing invalid authentication...');
    console.log('='.repeat(60));
    
    const invalidHeaders = {
        'X-API-Key': 'invalid-key-123',
        'Content-Type': 'application/json'
    };
    
    const data = {
        date: '2024-07-15',
        gmv: 9500000,
        users: 85000,
        marketing_cost: 175000
    };
    
    try {
        await axios.post(
            `${API_URL}/predict`,
            data,
            { headers: invalidHeaders }
        );
        
                console.log('Unexpected success - should have failed');
        return false;
        
    } catch (error) {
        if (error.response && error.response.status === 401) {
            console.log('Invalid auth correctly rejected');
            console.log(`  Error: ${error.response.data.error}`);
            return true;
        } else {
            console.log(`Unexpected error: ${error.message}`);
            return false;
        }
    }
}

/**
 * Run all tests
 */
async function main() {
    console.log('\n' + '='.repeat(60));
    console.log('Pod Forecasting API - Node.js Integration Demo');
    console.log('='.repeat(60));
    console.log(`\nAPI URL: ${API_URL}`);
    console.log(`API Key: ${API_KEY.length > 20 ? API_KEY.substring(0, 20) + '...' : API_KEY}`);
    
    // Run tests
    const tests = [
        { name: 'Health Check', func: testHealthCheck },
        { name: 'Single Prediction', func: testSinglePrediction },
        { name: 'Batch Prediction', func: testBatchPrediction },
        { name: 'Invalid Auth', func: testInvalidAuth },
    ];
    
    const results = [];
    
    for (const test of tests) {
        try {
            const result = await test.func();
            results.push({ name: test.name, result });
        } catch (error) {
            console.log(`\nTest '${test.name}' crashed: ${error.message}`);
            results.push({ name: test.name, result: false });
        }
    }
    
    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('Test Summary');
    console.log('='.repeat(60));
    
    results.forEach(({ name, result }) => {
    const status = result ? 'PASS' : 'FAIL';
    console.log(`${status} - ${name}`);
    });
    
    const passed = results.filter(r => r.result).length;
    const total = results.length;
    
    console.log('\n' + '='.repeat(60));
    console.log(`Results: ${passed}/${total} tests passed`);
    console.log('='.repeat(60));
    
    if (passed === total) {
        console.log('\nAll tests passed!');
        process.exit(0);
    } else {
        console.log(`\n${total - passed} test(s) failed`);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

// Export for use as module
module.exports = {
    testHealthCheck,
    testSinglePrediction,
    testBatchPrediction,
    testInvalidAuth
};