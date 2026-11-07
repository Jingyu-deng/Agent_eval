from langchain.tools import tool
from data import users_data
from dotenv import load_dotenv
import logging
from objects import PurchaseItem
from datetime import date, datetime


load_dotenv()  # Load environment variables from .env file

REFUND_WINDOW_DAYS = 30  # No refund may be issued more than this many days after purchase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  # Configure logging


def _parse_purchase_date(value: str) -> date | None:
    """Parse the dates stored in users_data, e.g. '2026-10-5' (not zero-padded)."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


@tool
def get_user_purchase_history(username: str) -> dict:
    """
    This tool retrieves the purchase history of a user based on their username.
    It checks if the username exists in the users_data dictionary and returns the corresponding purchase history.
    If the username is not found, it returns an empty dictionary.

    Args:
        username (str): The username of the user whose purchase history is to be retrieved.

    Returns:
        dict: A dictionary containing the user's purchase history if found, otherwise an empty dictionary.
    """
    logging.info(f"Retrieving purchase history for user: {username}")
    if users_data.get(username):
        return users_data[username]
    return {"status": "User not found"}  # Return an empty dictionary if the username is not found


@tool
def calculate_right_refund(username: str, items: list[PurchaseItem]) -> str:
    """
    This tool calculates the total refund amount for a user based on their purchase history and the items they want to return.
    It checks if the username exists in the users_data dictionary and calculates the refund amount for the specified items.
    Only items the user actually purchased and that are within the 30-day refund window are refundable.
    If the username is not found, it returns a message indicating that the user was not found.

    Args:
        username (str): The username of the user requesting a refund.
        items (list[PurchaseItem]): A list of PurchaseItem objects representing the items to be refunded.

    Returns:
        str: A message indicating the total refund amount and which items (if any) are not refundable.
    """
    logging.info(f"Calculating refund for user: {username} with items: {items}")
    if not users_data.get(username):
        return "User not found"

    history = users_data[username]
    today = date.today()
    total_refund = 0.0
    refundable = []
    not_refundable = []

    for item in items:
        item_name = item.name

        if item_name not in history:
            not_refundable.append(
                f"'{item_name}' was not purchased by {username}, so it cannot be refunded."
            )
            continue

        record = history[item_name]
        purchase_date = _parse_purchase_date(record.get("date"))
        if purchase_date is None:
            not_refundable.append(
                f"'{item_name}' has an unreadable purchase date ({record.get('date')}), so it cannot be refunded."
            )
            continue
        if purchase_date > today:
            not_refundable.append(
                f"'{item_name}' has a purchase date in the future ({purchase_date}), so it cannot be refunded."
            )
            continue

        age_days = (today - purchase_date).days
        if age_days > REFUND_WINDOW_DAYS:
            not_refundable.append(
                f"'{item_name}' was purchased on {purchase_date}, {age_days} days ago, "
                f"which is outside the {REFUND_WINDOW_DAYS}-day refund window."
            )
            continue

        total_refund += record["price"]
        refundable.append(f"'{item_name}': ${record['price']} (purchased {purchase_date})")

    lines = [f"Total refund amount: ${total_refund:g}"]
    if refundable:
        lines.append("Refundable items: " + "; ".join(refundable))
    if not_refundable:
        lines.append("Not refundable: " + " ".join(not_refundable))
    return "\n".join(lines)



@tool
def issue_refund_ticket(username: str, refund_amount: float) -> str:
    """
    This tool issues a refund ticket for a user based on their username and the specified refund amount.
    It checks if the username exists in the users_data dictionary and issues a refund ticket for the specified amount.
    If the username is not found, it returns a message indicating that the user was not found.

    Args:
        username (str): The username of the user requesting a refund.
        refund_amount (float): The amount to be refunded.

    Returns:
        str: A message indicating that a refund ticket has been issued or that the user was not found.
    """
    logging.info(f"Issuing refund ticket for user: {username} with amount: ${refund_amount}")
    if users_data.get(username):
        return f"A refund ticket with the amount ${refund_amount} has been issued and should be arriving soon to your bank account."
    return "User not found"  # Return a message if the username is not found


@tool
def get_today_date() -> str:
    """
    This tool retrieves the current date in the format YYYY-MM-DD.

    Returns:
        str: The current date in the format YYYY-MM-DD.
    """
    today = date.today()
    return today.strftime("%Y-%m-%d")

# print(get_today_date.invoke({}))  # Example invocation of the tool
# print(get_user_purchase_history.invoke("user123"))  # Example invocation of the tool
# print(calculate_right_refund.invoke({"username" :"user123", "items" : [PurchaseItem(name="notebook"), PurchaseItem(name="pen")]}))  # Example invocation of the tool
# print(issue_refund_ticket.invoke({"username" :"user123", "refund_amount" : 30}))  # Example invocation of the tool

