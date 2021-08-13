import pytest

from model_mommy import mommy
from rest_framework import status

from usaspending_api.awards.models import TransactionNormalized, TransactionFPDS, TransactionFABS
from usaspending_api.search.tests.data.utilities import setup_elasticsearch_test


url = "/api/v2/agency/{toptier_code}/sub_agency/{filter}"


@pytest.fixture
def transaction_search_1():

    # Submission
    dsws = mommy.make(
        "submissions.DABSSubmissionWindowSchedule",
        submission_reveal_date="2021-04-09",
        submission_fiscal_year=2021,
        submission_fiscal_month=7,
        submission_fiscal_quarter=3,
        is_quarter=False,
        period_start_date="2021-03-01",
        period_end_date="2021-04-01",
    )
    mommy.make("submissions.SubmissionAttributes", toptier_code="001", submission_window=dsws)
    mommy.make("submissions.SubmissionAttributes", toptier_code="002", submission_window=dsws)

    # Toptier and Awarding Agency
    toptier_agency_1 = mommy.make("references.ToptierAgency", toptier_code="001", name="Agency 1")
    toptier_agency_2 = mommy.make("references.ToptierAgency", toptier_code="002", name="Agency 2")
    subtier_agency_1 = mommy.make(
        "references.SubtierAgency",
        subtier_code="0001",
        name="Sub-Agency 1",
        abbreviation="A1",
    )
    subtier_agency_2 = mommy.make(
        "references.SubtierAgency", subtier_code="0002", name="Sub-Agency 2", abbreviation="A2"
    )
    awarding_agency_1 = mommy.make(
        "references.Agency", toptier_agency=toptier_agency_1, subtier_agency=subtier_agency_1, toptier_flag=True
    )
    awarding_agency_2 = mommy.make(
        "references.Agency", toptier_agency=toptier_agency_2, subtier_agency=subtier_agency_2, toptier_flag=True
    )
    mommy.make("references.Office", office_code="0001", office_name="Office 1")
    mommy.make("references.Office", office_code="0002", office_name="Office 2")

    # Awards
    award_contract = mommy.make(
        "awards.Award",
        category="contract",
        date_signed="2021-04-01",
    )
    award_idv = mommy.make(
        "awards.Award",
        category="idv",
        date_signed="2020-04-01",
    )
    award_grant = mommy.make(
        "awards.Award",
        category="grant",
        date_signed="2021-04-01",
    )
    award_loan = mommy.make(
        "awards.Award",
        category="loans",
        date_signed="2021-04-01",
    )
    award_dp = mommy.make(
        "awards.Award",
        category="direct payment",
        date_signed="2021-04-01",
    )

    contract_transaction = mommy.make(
        TransactionNormalized,
        award=award_contract,
        federal_action_obligation=101,
        action_date="2021-04-01",
        awarding_agency=awarding_agency_1,
        funding_agency=awarding_agency_1,
        is_fpds=True,
        type="A",
    )
    mommy.make(
        TransactionFPDS,
        transaction=contract_transaction,
        awarding_office_name="Office 1",
        awarding_office_code="0001",
        funding_office_name="Office 2",
        funding_office_code="0002",
    )

    idv_transaction = mommy.make(
        TransactionNormalized,
        award=award_idv,
        federal_action_obligation=102,
        action_date="2021-04-01",
        awarding_agency=awarding_agency_1,
        is_fpds=True,
        type="IDV_A",
    )
    mommy.make(
        TransactionFPDS,
        transaction=idv_transaction,
        awarding_office_name="Office 1",
        awarding_office_code="0001",
    )

    grant_transaction = mommy.make(
        TransactionNormalized,
        award=award_grant,
        federal_action_obligation=103,
        action_date="2021-04-01",
        awarding_agency=awarding_agency_1,
        is_fpds=False,
        type="04",
    )
    mommy.make(
        TransactionFABS,
        transaction=grant_transaction,
        awarding_office_name="Office 2",
        awarding_office_code="0002",
    )
    loan_transaction = mommy.make(
        TransactionNormalized,
        award=award_loan,
        federal_action_obligation=104,
        action_date="2021-04-01",
        awarding_agency=awarding_agency_1,
        is_fpds=False,
        type="09",
    )
    mommy.make(
        TransactionFABS,
        transaction=loan_transaction,
        awarding_office_name="Office 2",
        awarding_office_code="0002",
    )
    directpayment_transaction = mommy.make(
        TransactionNormalized,
        award=award_dp,
        federal_action_obligation=105,
        action_date="2021-04-01",
        awarding_agency=awarding_agency_1,
        is_fpds=False,
        type="10",
    )
    mommy.make(
        TransactionFABS,
        transaction=directpayment_transaction,
        awarding_office_name="Office 2",
        awarding_office_code="0002",
    )

    # Alternate Year
    idv_2 = mommy.make(
        TransactionNormalized,
        award=award_idv,
        federal_action_obligation=300,
        action_date="2020-04-01",
        awarding_agency=awarding_agency_1,
        is_fpds=True,
    )
    mommy.make(
        TransactionFPDS,
        transaction=idv_2,
        awarding_office_name="Office 1",
        awarding_office_code="0001",
    )

    # Alternate Agency
    idv_3 = mommy.make(
        TransactionNormalized,
        award=award_idv,
        federal_action_obligation=400,
        action_date="2021-04-01",
        awarding_agency=awarding_agency_2,
        is_fpds=True,
    )
    mommy.make(
        TransactionFPDS,
        transaction=idv_3,
        awarding_office_name="Office 2",
        awarding_office_code="0002",
    )


@pytest.mark.django_db
def test_all_categories(client, monkeypatch, transaction_search_1, elasticsearch_transaction_index):
    setup_elasticsearch_test(monkeypatch, elasticsearch_transaction_index)
    resp = client.get(url.format(toptier_code="001", filter="?fiscal_year=2021"))

    expected_results = [
        {
            "name": "Sub-Agency 1",
            "abbreviation": "A1",
            "total_obligations": 515,
            "transaction_count": 5,
            "new_award_count": 4,
            "children": [
                {"name": "Office 2", "total_obligations": 312.0, "transaction_count": 3, "new_award_count": 3},
                {"name": "Office 1", "total_obligations": 203.0, "transaction_count": 2, "new_award_count": 1},
            ],
        }
    ]
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["results"] == expected_results


@pytest.mark.django_db
def test_alternate_year(client, monkeypatch, transaction_search_1, elasticsearch_transaction_index):
    setup_elasticsearch_test(monkeypatch, elasticsearch_transaction_index)
    resp = client.get(url.format(toptier_code="001", filter="?fiscal_year=2020"))
    assert resp.status_code == status.HTTP_200_OK

    expected_results = [
        {
            "name": "Sub-Agency 1",
            "abbreviation": "A1",
            "total_obligations": 300.0,
            "transaction_count": 1,
            "new_award_count": 1,
            "children": [
                {"name": "Office 1", "total_obligations": 300.0, "transaction_count": 1, "new_award_count": 1}
            ],
        }
    ]
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["results"] == expected_results


@pytest.mark.django_db
def test_alternate_agency(client, monkeypatch, transaction_search_1, elasticsearch_transaction_index):
    setup_elasticsearch_test(monkeypatch, elasticsearch_transaction_index)
    resp = client.get(url.format(toptier_code="002", filter="?fiscal_year=2021"))
    assert resp.status_code == status.HTTP_200_OK

    expected_results = [
        {
            "name": "Sub-Agency 2",
            "abbreviation": "A2",
            "total_obligations": 400.0,
            "transaction_count": 1,
            "new_award_count": 0,
            "children": [
                {"name": "Office 2", "total_obligations": 400.0, "transaction_count": 1, "new_award_count": 0}
            ],
        }
    ]
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["results"] == expected_results


@pytest.mark.django_db
def test_award_types(client, monkeypatch, transaction_search_1, elasticsearch_transaction_index):
    setup_elasticsearch_test(monkeypatch, elasticsearch_transaction_index)
    resp = client.get(url.format(toptier_code="001", filter="?fiscal_year=2021&award_type_codes=[A]"))
    assert resp.status_code == status.HTTP_200_OK

    expected_results = [
        {
            "name": "Sub-Agency 1",
            "abbreviation": "A1",
            "total_obligations": 101.0,
            "transaction_count": 1,
            "new_award_count": 1,
            "children": [
                {"name": "Office 1", "total_obligations": 101.0, "transaction_count": 1, "new_award_count": 1}
            ],
        }
    ]
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["results"] == expected_results


@pytest.mark.django_db
def test_agency_types(client, monkeypatch, transaction_search_1, elasticsearch_transaction_index):
    setup_elasticsearch_test(monkeypatch, elasticsearch_transaction_index)
    resp = client.get(url.format(toptier_code="001", filter="?fiscal_year=2021&agency_type=funding"))
    assert resp.status_code == status.HTTP_200_OK

    expected_results = [
        {
            "name": "Sub-Agency 1",
            "abbreviation": "A1",
            "total_obligations": 101.0,
            "transaction_count": 1,
            "new_award_count": 1,
            "children": [
                {"name": "Office 2", "total_obligations": 101.0, "transaction_count": 1, "new_award_count": 1}
            ],
        }
    ]
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["results"] == expected_results


@pytest.mark.django_db
def test_invalid_agency(client, monkeypatch, transaction_search_1, elasticsearch_account_index):
    resp = client.get(url.format(toptier_code="XXX", filter="?fiscal_year=2021"))
    assert resp.status_code == status.HTTP_404_NOT_FOUND

    resp = client.get(url.format(toptier_code="999", filter="?fiscal_year=2021"))
    assert resp.status_code == status.HTTP_404_NOT_FOUND