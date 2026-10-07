const fs = require('fs');

const content = fs.readFileSync('quiz.html', 'utf8');

// Extract questions array
const match = content.match(/const questions = (\[[\s\S]*?\]);\s*let userAnswers/);
if (!match) {
    console.error("Could not find questions array in quiz.html");
    process.exit(1);
}

try {
    // evaluate the array
    const questions = eval(match[1]);
    console.log(`Found ${questions.length} questions in quiz.html.`);
    let allValid = true;
    questions.forEach((q, idx) => {
        if (!q.id || !q.title || !q.text || !Array.isArray(q.options) || q.options.length < 2) {
            console.error(`Question at index ${idx} is malformed:`, q);
            allValid = false;
        }
        const hasCorrect = q.options.some(o => o.correct === true);
        if (!hasCorrect) {
            console.error(`Question ${q.id} has no correct option!`);
            allValid = false;
        }
        if (!q.feedback || !q.feedback.correct) {
            console.warn(`Question ${q.id} (${q.title}) is missing feedback!`);
        }
    });
    if (allValid) {
        console.log("All questions are structurally valid!");
    } else {
        process.exit(1);
    }
} catch (e) {
    console.error("Syntax error parsing questions array:", e);
    process.exit(1);
}
