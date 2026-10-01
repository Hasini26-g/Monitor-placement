from adapter import Solver, dist_table
from benchkit.rng import Rng, derive_seed
import time


class MySolver(Solver):

    def solve(self, instance, submit_candidate):
        start = time.perf_counter()
        LIMIT = 4.98

        table = dist_table(instance)
        weights = instance.weights
        n = instance.size
        m = len(instance.sites)
        k = instance.k

        def elapsed():
            return time.perf_counter() - start

        def greedy(seed_sites=None, randomized=False):
            opened = []
            opened_set = set()
            nearest = [10**18] * n

            if seed_sites:
                for s in seed_sites:
                    opened.append(s)
                    opened_set.add(s)

                    for i in range(n):
                        d = table[i][s]
                        if d < nearest[i]:
                            nearest[i] = d

            while len(opened) < k:
                best_s = -1
                best_score = -1

                for s in range(m):
                    if s in opened_set:
                        continue

                    gain = 0

                    for i in range(n):
                        old = nearest[i]
                        new = table[i][s]

                        if new < old:
                            gain += weights[i] * (old - new)

                    if randomized:
                        noise = ((s * 1103515245 + len(opened) * 12345) & 1023)
                        score = gain * 1024 + noise
                    else:
                        score = gain

                    if score > best_score:
                        best_score = score
                        best_s = s

                opened.append(best_s)
                opened_set.add(best_s)

                for i in range(n):
                    d = table[i][best_s]
                    if d < nearest[i]:
                        nearest[i] = d

            return opened

        def build_state(opened):
            nearest = [10**18] * n
            second = [10**18] * n
            nearest_site = [-1] * n

            for i in range(n):
                row = table[i]

                best = 10**18
                second_best = 10**18
                best_site = -1

                for s in opened:
                    d = row[s]

                    if d < best:
                        second_best = best
                        best = d
                        best_site = s

                    elif d < second_best:
                        second_best = d

                nearest[i] = best
                second[i] = second_best
                nearest_site[i] = best_site

            cost = sum(
                weights[i] * nearest[i]
                for i in range(n)
            )

            return nearest, second, nearest_site, cost

        def improve(opened):
            opened = list(opened)
            opened_set = set(opened)

            nearest, second, nearest_site, cost = build_state(opened)

            while elapsed() < LIMIT:

                best_delta = 0
                best_out = -1
                best_in = -1

                for out in opened:

                    if elapsed() >= LIMIT:
                        break

                    for inn in range(m):

                        if inn in opened_set:
                            continue

                        delta = 0

                        for i in range(n):

                            old = nearest[i]

                            if nearest_site[i] == out:
                                base = second[i]
                            else:
                                base = old

                            nd = table[i][inn]

                            if nd < base:
                                delta += weights[i] * (nd - base)

                        if delta < best_delta:
                            best_delta = delta
                            best_out = out
                            best_in = inn

                if best_out == -1:
                    break

                idx = opened.index(best_out)
                opened[idx] = best_in

                opened_set.remove(best_out)
                opened_set.add(best_in)

                nearest, second, nearest_site, cost = build_state(opened)

                submit_candidate({
                    "sites": list(opened)
                })

            return opened, cost

        # First deterministic greedy solution
        opened = greedy()

        best_sites, best_cost = improve(opened)

        submit_candidate({
            "sites": list(best_sites)
        })

        # Randomized multi-start
        rng = Rng(
            derive_seed(
                "monitors-optimized",
                instance.digest
            )
        )

        attempts = 0

        while elapsed() < LIMIT:

            attempts += 1

            first = rng.randrange(m)

            if attempts <= 12:

                candidate = greedy(
                    seed_sites=[first],
                    randomized=True
                )

            else:

                second_start = rng.randrange(m)

                if second_start == first:
                    second_start = (second_start + 1) % m

                candidate = greedy(
                    seed_sites=[first, second_start],
                    randomized=True
                )

            candidate, candidate_cost = improve(candidate)

            if candidate_cost < best_cost:

                best_cost = candidate_cost
                best_sites = list(candidate)

                submit_candidate({
                    "sites": list(best_sites)
                })

        return {
            "sites": list(best_sites)
        }
