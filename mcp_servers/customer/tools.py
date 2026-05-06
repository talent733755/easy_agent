"""Mock tools for Customer MCP Server.

Provides mock data for 2 customers:
- Zhang Nu Shi (Zhang Lady)
- Li Nu Shi (Li Lady)
"""

from datetime import datetime, timedelta
from typing import Any


# Mock customer database
MOCK_CUSTOMERS: dict[str, dict[str, Any]] = {
    "zhang": {
        "customer_id": "C001",
        "name": "Zhang Nu Shi",
        "name_cn": "Zhang Lady",
        "phone": "138****1234",
        "membership_level": "gold",
        "registration_date": "2022-03-15",
        "total_spent": 12580.00,
        "preferences": ["skin care", "anti-aging", "moisturizing"],
        "skin_type": "dry",
        "allergies": ["alcohol"],
    },
    "li": {
        "customer_id": "C002",
        "name": "Li Nu Shi",
        "name_cn": "Li Lady",
        "phone": "139****5678",
        "membership_level": "silver",
        "registration_date": "2023-01-20",
        "total_spent": 5890.00,
        "preferences": ["whitening", "acne care"],
        "skin_type": "oily",
        "allergies": [],
    },
}

# Mock consumption history
MOCK_CONSUMPTION: dict[str, list[dict[str, Any]]] = {
    "C001": [
        {
            "date": "2024-01-15",
            "items": [
                {"product": "Moisturizing Cream 50ml", "price": 298.00, "quantity": 1},
                {"product": "Serum Vitamin C", "price": 398.00, "quantity": 1},
            ],
            "total": 696.00,
            "store": "Beijing Sanlitun",
            "staff": "Wang Xiaojie",
        },
        {
            "date": "2023-12-20",
            "items": [
                {"product": "Anti-aging Mask 5pcs", "price": 580.00, "quantity": 2},
            ],
            "total": 1160.00,
            "store": "Beijing Sanlitun",
            "staff": "Li Xiaojie",
        },
        {
            "date": "2023-10-08",
            "items": [
                {"product": "Sunscreen SPF50 100ml", "price": 258.00, "quantity": 1},
                {"product": "Cleansing Foam", "price": 158.00, "quantity": 1},
            ],
            "total": 416.00,
            "store": "Shanghai Nanjing Road",
            "staff": "Zhang Xiaojie",
        },
    ],
    "C002": [
        {
            "date": "2024-01-10",
            "items": [
                {"product": "Whitening Essence 30ml", "price": 458.00, "quantity": 1},
                {"product": "Acne Treatment Gel", "price": 168.00, "quantity": 2},
            ],
            "total": 794.00,
            "store": "Shenzhen Coco Park",
            "staff": "Chen Xiaojie",
        },
        {
            "date": "2023-11-15",
            "items": [
                {"product": "Oil Control Toner", "price": 198.00, "quantity": 1},
            ],
            "total": 198.00,
            "store": "Shenzhen Coco Park",
            "staff": "Liu Xiaojie",
        },
    ],
}

# Mock service history
MOCK_SERVICE_HISTORY: dict[str, list[dict[str, Any]]] = {
    "C001": [
        {
            "date": "2024-01-15",
            "service": "Deep Hydration Facial",
            "duration": "90 min",
            "price": 680.00,
            "therapist": "Wang Laoshi",
            "notes": "Customer prefers mild products, skin very dry",
            "satisfaction": 5,
        },
        {
            "date": "2023-12-20",
            "service": "Anti-aging Treatment",
            "duration": "120 min",
            "price": 1280.00,
            "therapist": "Li Laoshi",
            "notes": "Sensitive skin, avoid alcohol-based products",
            "satisfaction": 5,
        },
        {
            "date": "2023-09-10",
            "service": "Skin Analysis",
            "duration": "30 min",
            "price": 0,
            "therapist": "Zhang Laoshi",
            "notes": "Dry skin confirmed, recommended hydration routine",
            "satisfaction": 4,
        },
    ],
    "C002": [
        {
            "date": "2024-01-10",
            "service": "Acne Treatment Facial",
            "duration": "60 min",
            "price": 480.00,
            "therapist": "Chen Laoshi",
            "notes": "Oily T-zone, clogged pores on nose",
            "satisfaction": 4,
        },
        {
            "date": "2023-11-15",
            "service": "Oil Control Treatment",
            "duration": "45 min",
            "price": 320.00,
            "therapist": "Liu Laoshi",
            "notes": "Recommended weekly treatments",
            "satisfaction": 4,
        },
    ],
}


def get_customer(customer_name: str) -> dict[str, Any]:
    """Get customer information by name.

    Args:
        customer_name: Customer name (supports partial matching for 'Zhang', 'Li', etc.)

    Returns:
        Customer information dict, or empty dict if not found
    """
    customer_name_lower = customer_name.lower()

    # Try to match by key first (zhang, li)
    for key, customer in MOCK_CUSTOMERS.items():
        if key in customer_name_lower:
            return {
                "success": True,
                "data": customer,
            }

    # Try to match by name (Zhang Nu Shi, Li Nu Shi, etc.)
    for customer in MOCK_CUSTOMERS.values():
        if (
            customer_name_lower in customer["name"].lower()
            or customer_name_lower in customer["name_cn"].lower()
            or customer["name"].lower() in customer_name_lower
            or customer["name_cn"].lower() in customer_name_lower
        ):
            return {
                "success": True,
                "data": customer,
            }

    return {
        "success": False,
        "error": f"Customer '{customer_name}' not found",
        "data": None,
    }


def get_consumption(customer_id: str) -> dict[str, Any]:
    """Get customer consumption history.

    Args:
        customer_id: Customer ID (e.g., 'C001', 'C002')

    Returns:
        Consumption history dict with records list
    """
    records = MOCK_CONSUMPTION.get(customer_id, [])

    # Calculate total spent from records
    total_from_records = sum(r["total"] for r in records)

    return {
        "success": True,
        "customer_id": customer_id,
        "record_count": len(records),
        "total_amount": total_from_records,
        "records": records,
    }


def get_service_history(customer_id: str) -> dict[str, Any]:
    """Get customer service history.

    Args:
        customer_id: Customer ID (e.g., 'C001', 'C002')

    Returns:
        Service history dict with records list
    """
    records = MOCK_SERVICE_HISTORY.get(customer_id, [])

    # Calculate average satisfaction
    avg_satisfaction = (
        sum(r["satisfaction"] for r in records) / len(records)
        if records
        else 0
    )

    return {
        "success": True,
        "customer_id": customer_id,
        "record_count": len(records),
        "average_satisfaction": round(avg_satisfaction, 2),
        "records": records,
    }