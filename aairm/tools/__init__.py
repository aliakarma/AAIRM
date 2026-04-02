"""LangChain tool wrappers for AAIRM agent interactions.

Tools expose enterprise systems and external platforms through a common
interface layer.  Each tool is a LangChain ``Tool`` or ``StructuredTool``
that can be injected into any LangChain agent's tool list.

Modules
-------
inventory_tools  — inventory and ERP read/write operations
supplier_tools   — supplier catalogue queries and PO submission
logistics_tools  — carrier booking and tracking
erp_tools        — ERP / WMS integration helpers
"""
