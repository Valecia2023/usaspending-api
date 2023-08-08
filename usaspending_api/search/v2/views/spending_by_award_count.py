import copy
import logging

from sys import maxsize
from django.conf import settings
from django.db.models import Count
from rest_framework.response import Response
from rest_framework.views import APIView
from elasticsearch_dsl import Q

from usaspending_api.awards.v2.filters.sub_award import subaward_filter
from usaspending_api.awards.v2.lookups.lookups import all_award_types_mappings
from usaspending_api.common.api_versioning import api_transformations, API_TRANSFORM_FUNCTIONS
from usaspending_api.common.cache_decorator import cache_response
from usaspending_api.common.elasticsearch.search_wrappers import AwardSearch
from usaspending_api.common.exceptions import InvalidParameterException
from usaspending_api.common.helpers.generic_helper import (
    deprecated_district_field_in_location_object,
    get_generic_filters_message,
)
from usaspending_api.common.query_with_filters import QueryWithFilters
from usaspending_api.common.validator.award_filter import AWARD_FILTER_NO_RECIPIENT_ID
from usaspending_api.common.validator.pagination import PAGINATION
from usaspending_api.common.validator.tinyshield import TinyShield
from usaspending_api.search.filters.elasticsearch.filter import _QueryType
from usaspending_api.search.filters.time_period.decorators import NewAwardsOnlyTimePeriod
from usaspending_api.search.filters.time_period.query_types import AwardSearchTimePeriod

logger = logging.getLogger(__name__)


@api_transformations(api_version=settings.API_VERSION, function_list=API_TRANSFORM_FUNCTIONS)
class SpendingByAwardCountVisualizationViewSet(APIView):
    """This route takes award filters, and returns the number of awards in each award type.

    (Contracts, Loans, Grants, etc.)
    """

    endpoint_doc = "usaspending_api/api_contracts/contracts/v2/search/spending_by_award_count.md"

    @cache_response()
    def post(self, request):
        models = [
            {"name": "subawards", "key": "subawards", "type": "boolean", "default": False},
            {
                "name": "object_class",
                "key": "filter|object_class",
                "type": "array",
                "array_type": "text",
                "text_type": "search",
            },
            {
                "name": "program_activity",
                "key": "filter|program_activity",
                "type": "array",
                "array_type": "integer",
                "array_max": maxsize,
            },
        ]
        models.extend(copy.deepcopy(AWARD_FILTER_NO_RECIPIENT_ID))
        models.extend(copy.deepcopy(PAGINATION))
        self.original_filters = request.data.get("filters")
        json_request = TinyShield(models).block(request.data)
        subawards = json_request["subawards"]
        filters = json_request.get("filters", None)
        if filters is None:
            raise InvalidParameterException("Missing required request parameters: 'filters'")

        if "award_type_codes" in filters and "no intersection" in filters["award_type_codes"]:
            # "Special case": there will never be results when the website provides this value
            empty_results = {"contracts": 0, "idvs": 0, "grants": 0, "direct_payments": 0, "loans": 0, "other": 0}
            if subawards:
                empty_results = {"subcontracts": 0, "subgrants": 0}
            results = empty_results
        elif subawards:
            results = self.handle_subawards(filters)
        else:
            results = self.query_elasticsearch_for_prime_awards(filters)

        raw_response = {
            "results": results,
            "messages": get_generic_filters_message(
                self.original_filters.keys(), [elem["name"] for elem in AWARD_FILTER_NO_RECIPIENT_ID]
            ),
        }

        # Add filter field deprecation notices

        # TODO: To be removed in DEV-9966
        messages = raw_response.get("messages", [])
        deprecated_district_field_in_location_object(messages, self.original_filters)
        raw_response["messages"] = messages

        return Response(raw_response)

    @staticmethod
    def handle_subawards(filters: dict) -> dict:
        """Turn the filters into the result dictionary when dealing with Sub-Awards

        Note: Due to how the Django ORM joins to the awards table as an
        INNER JOIN, it is necessary to explicitly enforce the aggregations
        to only count Sub-Awards that are linked to a Prime Award.

        Remove the filter and update if we can move away from this behavior.
        """
        queryset = (
            subaward_filter(filters)
            .filter(award_id__isnull=False)
            .values("prime_award_group")
            .annotate(count=Count("broker_subaward_id"))
        )

        results = {}
        results["subgrants"] = sum([sub["count"] for sub in queryset if sub["prime_award_group"] == "grant"])
        results["subcontracts"] = sum([sub["count"] for sub in queryset if sub["prime_award_group"] == "procurement"])

        return results

    def query_elasticsearch_for_prime_awards(self, filters) -> list:
        filter_options = {}
        time_period_obj = AwardSearchTimePeriod(
            default_end_date=settings.API_MAX_DATE, default_start_date=settings.API_SEARCH_MIN_DATE
        )
        new_awards_only_decorator = NewAwardsOnlyTimePeriod(
            time_period_obj=time_period_obj, query_type=_QueryType.AWARDS
        )
        filter_options["time_period_obj"] = new_awards_only_decorator
        filter_query = QueryWithFilters.generate_awards_elasticsearch_query(filters, **filter_options)
        s = AwardSearch().filter(filter_query)

        s.aggs.bucket(
            "types",
            "filters",
            filters={category: Q("terms", type=types) for category, types in all_award_types_mappings.items()},
        )
        results = s.handle_execute()

        contracts = results.aggregations.types.buckets.contracts.doc_count
        idvs = results.aggregations.types.buckets.idvs.doc_count
        grants = results.aggregations.types.buckets.grants.doc_count
        direct_payments = results.aggregations.types.buckets.direct_payments.doc_count
        loans = results.aggregations.types.buckets.loans.doc_count
        other = results.aggregations.types.buckets.other_financial_assistance.doc_count

        response = {
            "contracts": contracts,
            "direct_payments": direct_payments,
            "grants": grants,
            "idvs": idvs,
            "loans": loans,
            "other": other,
        }
        return response
