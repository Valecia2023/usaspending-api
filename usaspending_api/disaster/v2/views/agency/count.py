from django.db.models import OuterRef, Q, Exists
from rest_framework.request import Request
from rest_framework.response import Response

from usaspending_api.common.cache_decorator import cache_response
from usaspending_api.disaster.v2.views.disaster_base import DisasterBase
from usaspending_api.financial_activities.models import FinancialAccountsByProgramActivityObjectClass
from usaspending_api.references.models import ToptierAgency


class AgencyCountViewSet(DisasterBase):
    """
    Obtain the count of Agencies related to supplied DEFC filter.
    """

    endpoint_doc = "usaspending_api/api_contracts/contracts/v2/disaster/agency/count.md"

    @cache_response()
    def post(self, request: Request) -> Response:
        filters = [
            Q(treasury_account__funding_toptier_agency=OuterRef("pk")),
            Q(disaster_emergency_fund__code__in=self.def_codes),
            self.all_closed_defc_submissions,
            Q(
                Q(obligations_incurred_by_program_object_class_cpe__gt=0)
                | Q(obligations_incurred_by_program_object_class_cpe__lt=0)
                | Q(gross_outlay_amount_by_program_object_class_cpe__gt=0)
                | Q(gross_outlay_amount_by_program_object_class_cpe__lt=0)
            ),
        ]
        count = (
            ToptierAgency.objects.annotate(
                include=Exists(FinancialAccountsByProgramActivityObjectClass.objects.filter(*filters).values("pk"))
            )
            .filter(include=True)
            .values("pk")
            .count()
        )
        return Response({"count": count})