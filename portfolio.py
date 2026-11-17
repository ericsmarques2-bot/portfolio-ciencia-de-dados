"""Três estudos reproduzíveis de ciência de dados com dados simulados."""

import argparse
import math
import random
from datetime import datetime, timedelta
from statistics import fmean


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def logistic_regression(rows):
    """Ajusta regressão logística com gradiente em lote e regularização L2."""
    weights = [0.0] * len(rows[0][0])
    prevalence = fmean(label for _, label in rows)
    weights[0] = math.log(prevalence / (1 - prevalence))
    for _ in range(600):
        gradient = [0.0] * len(weights)
        for features, label in rows:
            error = sigmoid(sum(w * x for w, x in zip(weights, features))) - label
            for index, feature in enumerate(features):
                gradient[index] += error * feature
        for index, value in enumerate(gradient):
            penalty = 0.001 * weights[index] if index else 0
            weights[index] -= 2 * (value / len(rows) + penalty)

    def predict(features):
        return sigmoid(sum(w * x for w, x in zip(weights, features)))

    return predict


def classification_rows(scenario):
    rng = random.Random(42)
    rows = []
    for index in range(6000):
        if scenario == "voos":
            weather, congestion, late_hour = [rng.random() for _ in range(3)]
            features = (1, weather, congestion, late_hour)
            # As três variáveis representam informações conhecidas antes da partida.
            log_odds = -3.2 + 2.6 * weather + 1.2 * congestion + 0.7 * late_hour
        else:
            amount, unusual_location, velocity = [rng.random() for _ in range(3)]
            features = (1, amount, unusual_location, velocity)
            # Os atributos são medidos no momento da autorização.
            log_odds = -8 + 2 * amount + 2.3 * unusual_location + 2.5 * velocity
        label = int(rng.random() < sigmoid(log_odds + 0.35 * index / 6000))
        rows.append((features, label))
    return rows


def evaluate_classifier(scenario):
    rows = classification_rows(scenario)
    train, validation, test = rows[:3600], rows[3600:4800], rows[4800:]
    predict = logistic_regression(train)

    def scored(rows_to_score):
        return sorted(((predict(x), y) for x, y in rows_to_score),
                      key=lambda item: item[0], reverse=True)

    ranked = scored(test)
    positives = sum(label for _, label in ranked)
    precision_at_100 = sum(label for _, label in ranked[:100]) / 100
    hits = 0
    precision_sum = 0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            hits += 1
            precision_sum += hits / rank
    average_precision = precision_sum / positives
    prevalence = positives / len(test)
    print(f"Teste cronológico: {len(test)} casos, {positives} positivos")
    print(f"Prevalência (referência aleatória): {prevalence:.1%}")
    print(f"Precisão nos 100 maiores escores: {precision_at_100:.1%}")
    print(f"Precisão média: {average_precision:.3f}")

    if scenario == "transacoes":
        # Seleciona na validação um limite que sinaliza cerca de 5% dos casos.
        threshold = scored(validation)[int(len(validation) * 0.05) - 1][0]
        alerts = [label for score, label in ranked if score >= threshold]
        precision = sum(alerts) / len(alerts) if alerts else 0
        recall = sum(alerts) / positives
        print(f"No limiar {threshold:.3f}: {len(alerts)} alertas, "
              f"precisão {precision:.1%}, recall {recall:.1%}")


def evaluate_bikes():
    rng = random.Random(42)
    start = datetime(2025, 1, 6)
    history = []
    errors = {"Semana anterior": [], "Média das últimas 4 semanas": []}
    for index in range(12 * 7 * 24):
        hour = start + timedelta(hours=index)
        peak = math.exp(-((hour.hour - 8) / 2.5) ** 2)
        peak += math.exp(-((hour.hour - 17) / 3) ** 2)
        weekday = hour.weekday() < 5
        expected = 10 + peak * (35 if weekday else 15) + 0.003 * index
        actual = max(0, round(expected + rng.gauss(0, 5)))
        if index >= 8 * 7 * 24:
            predictions = {
                "Semana anterior": history[index - 168],
                "Média das últimas 4 semanas": fmean(
                    history[index - offset * 168] for offset in range(1, 5)
                ),
            }
            for model, prediction in predictions.items():
                errors[model].append(abs(prediction - actual))
        history.append(actual)
    print("Teste em 4 semanas, com previsão horária usando apenas retiradas anteriores")
    for model, values in errors.items():
        print(f"{model}: MAE de {fmean(values):.2f} retiradas/hora")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projeto", choices=["voos", "transacoes", "bicicletas"])
    scenario = parser.parse_args().projeto
    if scenario == "bicicletas":
        evaluate_bikes()
    else:
        evaluate_classifier(scenario)


if __name__ == "__main__":
    main()
