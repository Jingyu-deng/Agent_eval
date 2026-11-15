from datetime import date, timedelta


def _days_ago(days: int) -> str:
    """Purchase date relative to today, so refund-window tests never go stale."""
    return (date.today() - timedelta(days=days)).isoformat()


users_data = {
    "user123": {
        "book": {
            "price": 23,
            "date": _days_ago(27)
        },
        "pen": {
            "price": 7,
            "date": _days_ago(29)
        }
    },
    "user124": {
        "notebook": {
            "price": 12,
            "date": _days_ago(25)
        }
    },
    "user125": {
        "pencil": {
            "price": 3,
            "date": _days_ago(17)
        },
        "ruler": {
            "price": 5,
            "date": _days_ago(37)
        }
    },
    "user126": {
        "backpack": {
            "price": 45,
            "date": _days_ago(29)
        },
        "water_bottle": {
            "price": 18,
            "date": _days_ago(98)
        },
        "lunch_box": {
            "price": 14,
            "date": _days_ago(21)
        }
    },
    "user127": {
        "eraser": {
            "price": 2,
            "date": _days_ago(21)
        }
    },
    "user128": {
        "stapler": {
            "price": 11,
            "date": _days_ago(21)
        },
        "highlighter": {
            "price": 4,
            "date": _days_ago(21)
        }
    },
    "user129": {
        "calculator": {
            "price": 29,
            "date": _days_ago(37)
        },
        "folder": {
            "price": 6,
            "date": _days_ago(6)
        },
        "usb_drive": {
            "price": 16,
            "date": _days_ago(96)
        }
    },
    "user130": {
        "headphones": {
            "price": 35,
            "date": _days_ago(21)
        }
    },
    "user131": {
        "desk_lamp": {
            "price": 28,
            "date": _days_ago(27)
        },
        "mouse": {
            "price": 20,
            "date": _days_ago(27)
        }
    },
    "user132": {
        "sketchbook": {
            "price": 17,
            "date": _days_ago(17)
        },
        "paintbrush": {
            "price": 9,
            "date": _days_ago(16)
        },
        "canvas": {
            "price": 25,
            "date": _days_ago(93)
        }
    }
}
