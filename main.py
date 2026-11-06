import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = ("passenger_count", "trip_distance", "fare_amount", "total_amount")
CSV_COLUMNS = (*NUMERIC_COLUMNS, "payment_type", "tpep_pickup_datetime")
PAYMENT_DESCRIPTIONS = {
	"1": "cartao de credito",
	"2": "dinheiro",
	"3": "sem cobranca",
	"4": "disputa",
	"5": "desconhecido",
	"6": "viagem anulada",
}


def read_chunks(csv_path, chunksize):
	return pd.read_csv(csv_path, usecols=CSV_COLUMNS, chunksize=chunksize)


def process_chunk(frame):
	result = {"rows": len(frame), "blocks": 1, "sums": {}, "counts": {}}

	for column in NUMERIC_COLUMNS:
		values = pd.to_numeric(frame[column], errors="coerce")
		result["sums"][column] = float(values.sum())
		result["counts"][column] = int(values.count())

	payments = frame["payment_type"].fillna("desconhecido").astype(str)
	result["payments"] = dict(Counter(payments))
	trip_days = frame["tpep_pickup_datetime"].astype("string").str.slice(0, 10)
	result["days"] = dict(Counter(trip_days.fillna("desconhecida")))
	amounts = pd.to_numeric(frame["total_amount"], errors="coerce").dropna()
	result["max_total_amount"] = float(amounts.max()) if not amounts.empty else None
	result["min_total_amount"] = float(amounts.min()) if not amounts.empty else None
	return result


def combine_results(total, partial):
	total["rows"] += partial["rows"]
	total["blocks"] += partial["blocks"]
	for column in NUMERIC_COLUMNS:
		total["sums"][column] += partial["sums"][column]
		total["counts"][column] += partial["counts"][column]
	total["payments"].update(partial["payments"])
	total["days"].update(partial["days"])
	for key, operation in (("max_total_amount", max), ("min_total_amount", min)):
		value = partial[key]
		if value is not None:
			current = total[key]
			total[key] = value if current is None else operation(current, value)


def empty_result():
	return {
		"rows": 0,
		"blocks": 0,
		"sums": {column: 0.0 for column in NUMERIC_COLUMNS},
		"counts": {column: 0 for column in NUMERIC_COLUMNS},
		"payments": Counter(),
		"days": Counter(),
		"max_total_amount": None,
		"min_total_amount": None,
	}


def run_sequential(csv_path, chunksize):
	result = empty_result()
	for frame in read_chunks(csv_path, chunksize):
		combine_results(result, process_chunk(frame))
	return result


def run_parallel(csv_path, chunksize, workers):
	result = empty_result()
	chunks = iter(read_chunks(csv_path, chunksize))
	max_in_flight = max(1, workers * 2)

	with ProcessPoolExecutor(max_workers=workers) as executor:
		pending = set()

		for _ in range(max_in_flight):
			try:
				pending.add(executor.submit(process_chunk, next(chunks)))
			except StopIteration:
				break

		while pending:
			completed, pending = wait(pending, return_when=FIRST_COMPLETED)
			for future in completed:
				combine_results(result, future.result())
				try:
					pending.add(executor.submit(process_chunk, next(chunks)))
				except StopIteration:
					pass

	return result


def benchmark_read(csv_path, chunksize, usecols):
	rows = 0
	start = time.perf_counter()
	for frame in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
		rows += len(frame)
	return rows, time.perf_counter() - start


def average(result, column):
	count = result["counts"][column]
	return result["sums"][column] / count if count else 0.0


def results_match(first, second):
	if (
		first["rows"] != second["rows"]
		or first["payments"] != second["payments"]
		or first["days"] != second["days"]
		or first["max_total_amount"] != second["max_total_amount"]
		or first["min_total_amount"] != second["min_total_amount"]
	):
		return False
	for column in NUMERIC_COLUMNS:
		if first["counts"][column] != second["counts"][column]:
			return False
		if not math.isclose(
			first["sums"][column], second["sums"][column], rel_tol=1e-9, abs_tol=1e-6
		):
			return False
	return True


def print_results(title, result, elapsed):
	print(f"\n{title} ({elapsed:.2f} s)")
	print(f"Viagens processadas: {result['rows']:,}")
	print(f"Distancia total: {result['sums']['trip_distance']:,.2f} milhas")
	mean_passengers = result['sums']['passenger_count'] / result['rows'] if result['rows'] else 0
	print(f"Passageiros medios por corrida: {mean_passengers:.4f}")
	print(f"Valor total: ${result['sums']['total_amount']:,.2f}")
	print(f"Maior valor de corrida: ${result['max_total_amount']:,.2f}")
	print(f"Menor valor de corrida: ${result['min_total_amount']:,.2f}")
	print("Viagens por tipo de pagamento:")
	for payment_type, count in sorted(result["payments"].items()):
		description = PAYMENT_DESCRIPTIONS.get(payment_type, "codigo nao catalogado")
		print(f"  {payment_type} ({description}): {count:,}")
	print("Top 5 dias com mais corridas:")
	for day, count in sorted(result["days"].items(), key=lambda item: (-item[1], item[0]))[:5]:
		print(f"  {day}: {count:,}")


def metrics_for_json(result):
	mean_passengers = result["sums"]["passenger_count"] / result["rows"] if result["rows"] else 0
	return {
		"quantidade_total_corridas": result["rows"],
		"distancia_total_milhas": round(result["sums"]["trip_distance"], 2),
		"valor_total_arrecadado": round(result["sums"]["total_amount"], 2),
		"media_passageiros_por_corrida": round(mean_passengers, 6),
		"maior_valor_corrida": result["max_total_amount"],
		"menor_valor_corrida": result["min_total_amount"],
		"corridas_por_forma_pagamento": {
			code: {
				"descricao": PAYMENT_DESCRIPTIONS.get(code, "codigo nao catalogado"),
				"quantidade": count,
			}
			for code, count in sorted(result["payments"].items())
		},
		"top_5_dias": [
			{"data": day, "corridas": count}
			for day, count in sorted(result["days"].items(), key=lambda item: (-item[1], item[0]))[:5]
		],
	}


def save_artifacts(
	args, sequential, parallel, sequential_time, parallel_time, selected_read, full_read, run_started
):
	base_dir = Path(__file__).resolve().parent
	results_dir = base_dir / "resultados"
	logs_dir = base_dir / "logs"
	results_dir.mkdir(exist_ok=True)
	logs_dir.mkdir(exist_ok=True)

	metrics = metrics_for_json(sequential)
	metrics["execucao"] = {
		"arquivo": str(args.csv.resolve()),
		"chunksize": args.chunksize,
		"numero_de_blocos": sequential["blocks"],
		"tempo_sequencial_segundos": round(sequential_time, 4),
		"tempo_paralelo_segundos": round(parallel_time, 4),
		"workers": args.workers,
		"resultados_equivalentes": results_match(sequential, parallel),
	}
	(results_dir / "resultados_parte_a.json").write_text(
		json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
	)

	all_columns = len(pd.read_csv(args.csv, nrows=0).columns)
	selected_columns = len(CSV_COLUMNS)
	read_gain = (full_read - selected_read) / full_read * 100 if full_read else 0.0
	log_lines = [
		f"Inicio: {run_started.isoformat(timespec='seconds')}",
		f"Arquivo: {args.csv.resolve()}",
		f"Sistema operacional: {platform.platform()}",
		f"Processador: {platform.processor() or 'nao informado'}",
		f"Nucleos logicos: {os.cpu_count()}",
		f"Python: {sys.version.split()[0]}",
		f"pandas: {pd.__version__}",
		f"Chunksize: {args.chunksize}",
		f"Blocos processados: {sequential['blocks']}",
		f"Versao sequencial: {sequential_time:.2f} s",
		f"Versao paralela: {parallel_time:.2f} s",
		f"Workers: {args.workers}",
		f"Resultados equivalentes: {results_match(sequential, parallel)}",
		f"Leitura selecionada ({selected_columns} colunas): {selected_read:.2f} s",
		f"Leitura completa ({all_columns} colunas): {full_read:.2f} s",
		f"Reducao no tempo de leitura com usecols: {read_gain:.2f}%",
		f"Fim: {datetime.now().astimezone().isoformat(timespec='seconds')}",
	]
	(logs_dir / "parte_a.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Compara processamento sequencial e paralelo de um CSV em blocos."
	)
	parser.add_argument(
		"csv",
		nargs="?",
		type=Path,
		default=Path(__file__).with_name("yellow_tripdata_2016-02.csv"),
		help="caminho do CSV (padrao: yellow_tripdata_2016-02.csv ao lado do script)",
	)
	parser.add_argument("--chunksize", type=int, default=100_000, help="linhas por bloco")
	parser.add_argument(
		"--workers",
		type=int,
		default=max(1, min(4, os.cpu_count() or 1)),
		help="numero de processos paralelos (padrao: ate 4)",
	)
	args = parser.parse_args()
	if args.chunksize < 1:
		parser.error("--chunksize deve ser maior que zero")
	if args.workers < 1:
		parser.error("--workers deve ser maior que zero")
	if not args.csv.is_file():
		parser.error(f"arquivo CSV nao encontrado: {args.csv}")
	return args


def main():
	args = parse_args()
	run_started = datetime.now().astimezone()

	start = time.perf_counter()
	sequential = run_sequential(args.csv, args.chunksize)
	sequential_time = time.perf_counter() - start

	start = time.perf_counter()
	parallel = run_parallel(args.csv, args.chunksize, args.workers)
	parallel_time = time.perf_counter() - start

	selected_rows, selected_read = benchmark_read(args.csv, args.chunksize, CSV_COLUMNS)
	full_rows, full_read = benchmark_read(args.csv, args.chunksize, None)
	if selected_rows != sequential["rows"] or full_rows != sequential["rows"]:
		raise RuntimeError("A quantidade de linhas divergiu durante os benchmarks de leitura")

	print(f"Arquivo: {args.csv}")
	print(f"Blocos: {args.chunksize:,} linhas | Workers: {args.workers}")
	print_results("Versao sequencial", sequential, sequential_time)
	print_results("Versao paralela", parallel, parallel_time)

	speedup = sequential_time / parallel_time if parallel_time else float("inf")
	print(f"\nSpeedup (sequencial / paralelo): {speedup:.2f}x")
	print(f"Resultados equivalentes: {'sim' if results_match(sequential, parallel) else 'nao'}")
	print(f"Leitura com {len(CSV_COLUMNS)} colunas: {selected_read:.2f} s")
	print(f"Leitura com todas as colunas: {full_read:.2f} s")
	read_gain = (full_read - selected_read) / full_read * 100 if full_read else 0.0
	print(f"Reducao no tempo de leitura com usecols: {read_gain:.2f}%")
	save_artifacts(
		args, sequential, parallel, sequential_time, parallel_time, selected_read, full_read, run_started
	)
	print("JSON salvo em resultados/resultados_parte_a.json")
	print("Log salvo em logs/parte_a.log")
	print(
		"Paralelismo pode reduzir o tempo de calculo, mas a criacao dos processos, "
		"a transferencia dos blocos e a leitura do disco geram overhead. Blocos maiores "
		"reduzem chamadas e transferencias, mas aumentam o uso de memoria; mais workers "
		"tambem nao garantem ganho quando disco ou CPU ja estao saturados."
	)


if __name__ == "__main__":
	main()
