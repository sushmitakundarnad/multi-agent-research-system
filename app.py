import time
import streamlit as st

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer → Critic")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "timings" not in st.session_state:
    st.session_state.timings = {}


# ----------------------------------------------------------------------------
# Pipeline (same steps as pipeline.py, but reports progress to the UI)
# ----------------------------------------------------------------------------
def run_pipeline_with_ui(topic: str) -> dict:
    state = {}
    timings = {}

    with st.status("Running research pipeline...", expanded=True) as status:
        # Step 1 - Search agent
        st.write("**Step 1/4** - Search agent is working...")
        t0 = time.time()
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content
        timings["Search"] = time.time() - t0
        st.write(f"✅ Search complete ({timings['Search']:.1f}s)")

        # Step 2 - Reader agent
        st.write("**Step 2/4** - Reader agent is scraping top resources...")
        t0 = time.time()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
        timings["Reader"] = time.time() - t0
        st.write(f"✅ Scraping complete ({timings['Reader']:.1f}s)")

        # Step 3 - Writer chain
        st.write("**Step 3/4** - Writer is drafting the report...")
        t0 = time.time()
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        timings["Writer"] = time.time() - t0
        st.write(f"✅ Report drafted ({timings['Writer']:.1f}s)")

        # Step 4 - Critic chain
        st.write("**Step 4/4** - Critic is reviewing the report...")
        t0 = time.time()
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        timings["Critic"] = time.time() - t0
        st.write(f"✅ Review complete ({timings['Critic']:.1f}s)")

        status.update(
            label=f"Pipeline complete in {sum(timings.values()):.1f}s",
            state="complete",
            expanded=False,
        )

    st.session_state.timings = timings
    return state


def to_text(value) -> str:
    """Chain outputs may be str or message objects; normalise to text."""
    if isinstance(value, str):
        return value
    return getattr(value, "content", str(value))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "1. **Search Agent** finds recent sources\n"
        "2. **Reader Agent** scrapes the best URL\n"
        "3. **Writer** drafts a report\n"
        "4. **Critic** reviews the report"
    )
    st.divider()
    st.subheader("Examples")
    examples = [
        "Latest advances in solid-state batteries",
        "Impact of AI on drug discovery",
        "State of quantum computing in 2026",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.topic_input = ex
    st.divider()
    if st.button("🗑️ Clear results", use_container_width=True):
        st.session_state.state = None
        st.session_state.timings = {}
        st.rerun()

# ----------------------------------------------------------------------------
# Main input
# ----------------------------------------------------------------------------
topic = st.text_input(
    "Research topic",
    key="topic_input",
    placeholder="e.g. Latest advances in solid-state batteries",
)

run = st.button("🚀 Run research", type="primary", disabled=not topic.strip())

if run:
    st.session_state.topic = topic.strip()
    try:
        st.session_state.state = run_pipeline_with_ui(st.session_state.topic)
    except Exception as e:
        st.session_state.state = None
        st.error(f"Pipeline failed: {e}")
        st.exception(e)

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
state = st.session_state.state
if state:
    st.subheader(f"Results: {st.session_state.topic}")

    if st.session_state.timings:
        cols = st.columns(len(st.session_state.timings))
        for col, (name, secs) in zip(cols, st.session_state.timings.items()):
            col.metric(name, f"{secs:.1f}s")

    tab_report, tab_feedback, tab_search, tab_scrape = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "🕸️ Scraped Content"]
    )

    report_text = to_text(state["report"])
    feedback_text = to_text(state["feedback"])

    with tab_report:
        st.markdown(report_text)
        st.download_button(
            "⬇️ Download report (.md)",
            data=report_text,
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(feedback_text)

    with tab_search:
        st.markdown(to_text(state["search_results"]))

    with tab_scrape:
        st.markdown(to_text(state["scraped_content"]))