from src.schedule_generator.main import JobShopProblem
from src.production_orders import parse_data


def test_p1():
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    p1_jobs = [j for j in jssp.jobs if j.production_order_nr == "P1"]
    assert len(p1_jobs) == 8
    p1_amounts = [j.amount for j in p1_jobs if len(j.available_machines) == 1]
    assert sum(p1_amounts) == 16000
    assert all([j.station_settings["taste"] == "cola" for j in p1_jobs])


def test_p2():
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    p2_jobs = [j for j in jssp.jobs if j.production_order_nr == "P2"]
    assert len(p2_jobs) == 10
    p2_amounts = [j.amount for j in p2_jobs if len(j.available_machines) == 1]
    assert sum(p2_amounts) == 18000
    assert all([j.station_settings["taste"] == "cola" for j in p2_jobs])


def test_p3():
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    p3_jobs = [j for j in jssp.jobs if j.production_order_nr == "P3"]
    assert len(p3_jobs) == 4
    p3_amounts = [j.amount for j in p3_jobs if len(j.available_machines) == 1]
    assert sum(p3_amounts) == 6000
    assert all([j.station_settings["taste"] == "cola" for j in p3_jobs])


def test_p4():
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    p4_jobs = [j for j in jssp.jobs if j.production_order_nr == "P4"]
    assert len(p4_jobs) == 2
    p4_amounts = [j.amount for j in p4_jobs if len(j.available_machines) == 1]
    assert sum(p4_amounts) == 3750
    assert all([j.station_settings["taste"] == "fanta" for j in p4_jobs])


def test_p5():
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    p5_jobs = [j for j in jssp.jobs if j.production_order_nr == "P5"]
    assert len(p5_jobs) == 12
    p5_amounts = [j.amount for j in p5_jobs if len(j.available_machines) == 1]
    assert sum(p5_amounts) == 22000
    assert all([j.station_settings["taste"] == "fanta" for j in p5_jobs])


def test_p6():
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    p6_jobs = [j for j in jssp.jobs if j.production_order_nr == "P6"]
    assert len(p6_jobs) == 10
    p6_amounts = [j.amount for j in p6_jobs if len(j.available_machines) == 1]
    assert sum(p6_amounts) == 16500
    assert all([j.station_settings["taste"] == "fanta" for j in p6_jobs])


def test_p7():
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    p7_jobs = [j for j in jssp.jobs if j.production_order_nr == "P7"]
    assert len(p7_jobs) == 2
    p7_amounts = [j.amount for j in p7_jobs if len(j.available_machines) == 1]
    assert sum(p7_amounts) == 3000
    assert all([j.station_settings["taste"] == "cola" for j in p7_jobs])
