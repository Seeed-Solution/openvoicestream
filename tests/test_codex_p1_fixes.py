"""Regression test suite for Codex P1 review fixes.

Validates:
1. Gate protection in verify_asr_and_push.py: Airtight AST containment proving that
   ALL remote mkdir operations and ALL push loops are strictly enclosed within the
   `else` (gate pass) branch of `if not all_gates_passed:` and completely unreachable
   when gates fail.
2. Negative controls proving that:
   - An unnested/dedented push loop is rejected.
   - A guarded loop plus an additional unguarded loop is rejected.
   - A guarded loop with an unguarded remote mkdir is rejected.
3. Definition of `base_weights` in `run_phase2_training` before its use in L2-SP.
4. Definition of `base_proj_weight` and `base_proj_bias` in `main()` before the
   Phase 1 winning checkpoint packaging logic.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_push_gate_containment(tree: ast.AST) -> bool:
    """Verifies that within the AST:

    1. A gate node `if not all_gates_passed:` exists.
    2. At least one push loop (iterating over `generated_files`) exists.
    3. At least one remote mkdir assignment (`mkdir_cmd = ...`) exists.
    4. EVERY push loop across the entire AST is strictly contained within the gate's `orelse` subtree.
    5. EVERY remote mkdir assignment across the entire AST is strictly contained within the gate's `orelse` subtree.
    """
    gate_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
                if isinstance(node.test.operand, ast.Name) and node.test.operand.id == "all_gates_passed":
                    gate_node = node
                    break

    if gate_node is None:
        return False

    orelse_nodes = set()
    for child in gate_node.orelse:
        for sub in ast.walk(child):
            orelse_nodes.add(id(sub))

    # Identify all push loops across the entire AST
    all_push_loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "generated_files"
    ]
    if not all_push_loops:
        return False

    # Check that EVERY push loop belongs to orelse
    for loop in all_push_loops:
        if id(loop) not in orelse_nodes:
            return False

    # Identify all remote mkdir operations across the entire AST
    all_mkdir_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "mkdir_cmd":
                    all_mkdir_nodes.append(node)

    if not all_mkdir_nodes:
        return False

    # Check that EVERY mkdir assignment belongs to orelse
    for mk in all_mkdir_nodes:
        if id(mk) not in orelse_nodes:
            return False

    return True


class TestCodexP1Fixes(unittest.TestCase):
    def test_verify_asr_and_push_gate_nesting_ast(self):
        script_path = REPO_ROOT / "tools" / "voice_pipeline" / "verify_asr_and_push.py"
        with open(script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(script_path))

        self.assertTrue(
            check_push_gate_containment(tree),
            "CRITICAL: Remote mkdir and push loops must all be strictly nested inside the `else` branch of `if not all_gates_passed:`",
        )

    def test_negative_control_dedented_push_loop(self):
        """Negative control: verify that unnested/dedented push loop is rejected."""
        buggy_code = """
def run_evaluation():
    if not all_gates_passed:
        print("[-] CANNOT PUSH TO MACBOOK: Quality gates failed! Delivery blocked.")
        fleet_success = False
    else:
        print(f"Remote Destination: macbook:{MACBOOK_DEST_DIR}")

    mkdir_cmd = [FLEET_BIN, "exec", "macbook", "mkdir", "-p", MACBOOK_DEST_DIR]
    for local_path, fname in generated_files:
        push_cmd = [FLEET_BIN, "push", "macbook", local_path, target_remote]
"""
        self.assertFalse(
            check_push_gate_containment(ast.parse(buggy_code)),
            "Negative control failed: must reject unnested push loop!",
        )

    def test_negative_control_duplicate_unguarded_push_loop(self):
        """Negative control: verify that a guarded loop plus an additional unguarded loop is rejected."""
        buggy_code = """
def run_evaluation():
    if not all_gates_passed:
        pass
    else:
        mkdir_cmd = [FLEET_BIN, "exec", "macbook", "mkdir", "-p", MACBOOK_DEST_DIR]
        for local_path, fname in generated_files:
            push_cmd = [FLEET_BIN, "push", "macbook", local_path, target_remote]
    for local_path, fname in generated_files:
        push_cmd = [FLEET_BIN, "push", "macbook", local_path, target_remote]
"""
        self.assertFalse(
            check_push_gate_containment(ast.parse(buggy_code)),
            "Negative control failed: must reject duplicate unguarded push loop!",
        )

    def test_negative_control_unguarded_mkdir(self):
        """Negative control: verify that a guarded push loop with an unguarded remote mkdir is rejected."""
        buggy_code = """
def run_evaluation():
    if not all_gates_passed:
        pass
    else:
        for local_path, fname in generated_files:
            push_cmd = [FLEET_BIN, "push", "macbook", local_path, target_remote]
    mkdir_cmd = [FLEET_BIN, "exec", "macbook", "mkdir", "-p", MACBOOK_DEST_DIR]
"""
        self.assertFalse(
            check_push_gate_containment(ast.parse(buggy_code)),
            "Negative control failed: must reject unguarded remote mkdir!",
        )

    def test_phase2_base_weights_assigned_before_loop(self):
        script_path = REPO_ROOT / "tools" / "voice_pipeline" / "finetune_official_matcha_v3.py"
        with open(script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(script_path))

        phase2_fn = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run_phase2_training":
                phase2_fn = node
                break

        self.assertIsNotNone(phase2_fn, "run_phase2_training function not found")

        assigned_base_weights = False
        used_base_weights = False

        for stmt in phase2_fn.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "base_weights":
                        assigned_base_weights = True
            if isinstance(stmt, ast.For):
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name) and n.id == "base_weights":
                        used_base_weights = True
                        self.assertTrue(
                            assigned_base_weights,
                            "base_weights was accessed in loop before being assigned in run_phase2_training",
                        )

        self.assertTrue(assigned_base_weights, "base_weights assignment not found in run_phase2_training")
        self.assertTrue(used_base_weights, "base_weights usage not found in run_phase2_training")

    def test_phase1_winner_has_base_proj_assigned(self):
        script_path = REPO_ROOT / "tools" / "voice_pipeline" / "finetune_official_matcha_v3.py"
        with open(script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(script_path))

        main_fn = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                main_fn = node
                break

        self.assertIsNotNone(main_fn, "main function not found")

        assigned_proj_weight = False
        assigned_proj_bias = False

        for stmt in main_fn.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "base_proj_weight":
                            assigned_proj_weight = True
                        if target.id == "base_proj_bias":
                            assigned_proj_bias = True
            if isinstance(stmt, ast.If):
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name) and n.id == "base_proj_weight":
                        self.assertTrue(
                            assigned_proj_weight,
                            "base_proj_weight was accessed in winning checkpoint logic before being assigned",
                        )
                    if isinstance(n, ast.Name) and n.id == "base_proj_bias":
                        self.assertTrue(
                            assigned_proj_bias,
                            "base_proj_bias was accessed in winning checkpoint logic before being assigned",
                        )

        self.assertTrue(assigned_proj_weight, "base_proj_weight assignment not found in main()")
        self.assertTrue(assigned_proj_bias, "base_proj_bias assignment not found in main()")


if __name__ == "__main__":
    unittest.main()
