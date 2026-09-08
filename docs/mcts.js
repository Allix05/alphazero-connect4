// PUCT Monte Carlo Tree Search guided by the ONNX policy/value network,
// ported from alphazero/mcts.py. Runs entirely in the browser.
const AZMCTS = (() => {
  const C_PUCT = 1.5;

  class Node {
    constructor(board, parent = null, prior = 0) {
      this.board = board;
      this.parent = parent;
      this.children = new Map(); // action -> Node
      this.prior = prior;
      this.visitCount = 0;
      this.valueSum = 0;
      this.isExpanded = false;
    }

    get value() {
      return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
    }

    ucbScore(totalVisits) {
      const exploration = C_PUCT * this.prior * Math.sqrt(totalVisits) / (1 + this.visitCount);
      return -this.value + exploration;
    }
  }

  function selectChild(node) {
    let totalVisits = 0;
    for (const child of node.children.values()) totalVisits += child.visitCount;
    let best = null, bestScore = -Infinity;
    for (const child of node.children.values()) {
      const score = child.ucbScore(totalVisits);
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }

  function expand(node, probs, valid) {
    for (let action = 0; action < C4.COLS; action++) {
      if (valid[action]) {
        const childBoard = C4.nextState(node.board, action);
        node.children.set(action, new Node(childBoard, node, probs[action]));
      }
    }
    node.isExpanded = true;
  }

  function backpropagate(path, leafValue) {
    let value = leafValue;
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].visitCount += 1;
      path[i].valueSum += value;
      value = -value;
    }
  }

  // `evaluate(board)` must return a Promise<{probs: Float32Array(7), value: number}>
  // (already softmax'd policy probabilities, pre-masking).
  async function run(evaluate, rootBoard, numSimulations) {
    const root = new Node(rootBoard);
    const valid = C4.validMoves(rootBoard);

    let { probs } = await evaluate(rootBoard);
    probs = maskAndNormalize(probs, valid);
    expand(root, probs, valid);

    for (let i = 0; i < numSimulations; i++) {
      let node = root;
      const path = [node];
      while (node.isExpanded && node.children.size > 0) {
        node = selectChild(node);
        path.push(node);
      }

      const outcome = C4.gameEnded(node.board);
      let value;
      if (outcome === null) {
        const evalResult = await evaluate(node.board);
        const nodeValid = C4.validMoves(node.board);
        const nodeProbs = maskAndNormalize(evalResult.probs, nodeValid);
        expand(node, nodeProbs, nodeValid);
        value = evalResult.value;
      } else {
        value = Math.abs(outcome) < 1e-3 ? 0.0 : -outcome;
      }
      backpropagate(path, value);
    }

    const visitCounts = new Float32Array(C4.COLS);
    for (const [action, child] of root.children) visitCounts[action] = child.visitCount;
    const total = visitCounts.reduce((a, b) => a + b, 0);
    if (total > 0) for (let i = 0; i < visitCounts.length; i++) visitCounts[i] /= total;
    return visitCounts;
  }

  function maskAndNormalize(probs, valid) {
    const masked = new Float32Array(probs.length);
    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
      if (valid[i]) {
        masked[i] = probs[i];
        sum += probs[i];
      }
    }
    if (sum > 0) {
      for (let i = 0; i < masked.length; i++) masked[i] /= sum;
    } else {
      const nValid = valid.filter(Boolean).length;
      for (let i = 0; i < masked.length; i++) if (valid[i]) masked[i] = 1 / nValid;
    }
    return masked;
  }

  return { run };
})();
