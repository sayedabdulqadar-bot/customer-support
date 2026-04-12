from env.models import GraderResult

def grade(task_id, obs):
    score = 0.0

    if obs.kb_searched:
        score += 0.2
    if obs.empathized:
        score += 0.2
    if obs.solution_offered:
        score += 0.3
    if obs.done:
        score += 0.3

    # penalties
    if obs.escalated:
        score *= 0.7

    if obs.turn > 6:
        score *= 0.9

    score = min(max(score, 0.001), 0.999)

    return GraderResult(
        task_id=task_id,
        score=round(score, 3),
        passed=score > 0.5,
        breakdown={}
    )
