# Validation Error Analysis
- Total validation records: 4429
- Total errors: 250
- Error rate: 0.0564

## Top confusion pairs
- Write -> Read: 74
- Read -> Write: 62
- Execute -> Read: 28
- Execute -> Write: 21
- Read -> Execute: 12
- Other -> Read: 12
- Write -> Execute: 11
- Financial -> Write: 7
- Destructive -> Read: 5
- Destructive -> Write: 5

## Recurring failure modes
1. Majority-class pull: 216 errors were predicted as Read/Write. Improvement: moderate class-weighted loss.
2. Write vs Execute ambiguity: 32 direct confusions. Improvement: emphasize state change versus actual code/command execution.
3. Minority-class weakness: 73 errors came from Execute/Financial/Other. Improvement: targeted weighting and minority-focused examples.

## 20 errors for manual review
1. `crypto_dtd` | true=Execute | pred=Read | conf=0.9990 | Get the Distance-to-Default (DtD) score for a crypto token. DtD measures distance-to-default on a 0-5 scale (5=healthy, 0=imminent collapse). Returns 7 signal scores (Liquidity, Holders, Resilience, Fundamental, Contagio
2. `managed_batch_translation_workflow` | true=Execute | pred=Write | conf=0.8851 | Execute managed batch translation workflow with comprehensive monitoring and analytics.  This workflow provides complete batch translation lifecycle management: 1. Pre-validates language pairs and terminologies 2. Starts
3. `cancel_metadata_transfer_job` | true=Financial | pred=Destructive | conf=0.9051 | Cancel a metadata transfer job.  Args:     metadata_transfer_job_id: The ID of the metadata transfer job to cancel     region: AWS region (default: us-east-1)  Returns:     Dictionary containing cancellation response
4. `create_metadata_transfer_job` | true=Financial | pred=Write | conf=0.9004 | Create a new metadata transfer job for bulk import/export operations between S3 and IoT SiteWise.  This tool provides a user-friendly way to set up metadata transfer jobs with support for bulk export using IoT SiteWise s
5. `lambda_function` | true=Other | pred=Read | conf=0.9875 | Tool for invoking a specific AWS Lambda function with parameters.
6. `mcp_ado_pipelines_update_build_stage` | true=Execute | pred=Write | conf=0.9607 | Update a build stage (cancel, retry, or run)
7. `brand.scan` | true=Execute | pred=Read | conf=0.9708 | Run AI brand visibility scan across major LLM providers (async, poll with brand.scan.get).
8. `scout.reddit` | true=Execute | pred=Read | conf=0.6702 | Run Reddit scout analysis. Returns processing status — poll scout.reddit.result for results.
9. `scout.reddit.result` | true=Execute | pred=Read | conf=0.7732 | Get Reddit scout run status and results by run ID.
10. `scout.x.result` | true=Execute | pred=Read | conf=0.8094 | Get X scout run status and results by run ID.
11. `understand_image` | true=Other | pred=Read | conf=0.9972 | Analyze images with AI vision — screenshots, diagrams, UI mockups, error messages, charts, photos. Provide a URL or local file path to the image.
12. `dolores_new_memecoins` | true=Execute | pred=Read | conf=0.9982 | Fetch newly launched memecoins on PumpFun from DexScreener live data. Returns real-time token info including price, market cap, volume, and contract address.
13. `inverspec_phase_1_architecture` | true=Execute | pred=Read | conf=0.9364 | Returns the Phase 1 prompt template for high-level architecture and request-flow mapping. Run after Phase 0 inventory is complete.
14. `inverspec_phase_7_maintenance` | true=Execute | pred=Read | conf=0.9944 | Returns the Phase 7 prompt template for partial updates to an existing technical specification after code changes. Maps changed file patterns to the right spec phase and rewrites only affected sections. Requires a comple
15. `aide_init` | true=Execute | pred=Read | conf=0.7419 | Bootstrap the AIDE development environment into a project. Returns structured JSON for agent consumption — not prose.  The tool uses a two-call pattern for progressive disclosure:  **First call (no `category` param):** R
16. `aide_upgrade` | true=Execute | pred=Write | conf=0.6483 | Compare the AIDE methodology artifacts in this project against the canonical versions and return structured JSON results grouped by category. Use this when the user asks to update AIDE, sync AIDE, refresh AIDE, check for
17. `argument_validation` | true=Execute | pred=Read | conf=0.9977 | Analyse arguments for logical fallacies
18. `code_review_deep` | true=Execute | pred=Read | conf=0.9983 | Review code for security, performance, and quality
19. `create_function_schema` | true=Execute | pred=Write | conf=0.9969 | Generate JSON Schema for function calling
20. `creative_ideation` | true=Execute | pred=Read | conf=0.9420 | Generate creative ideas with feasibility analysis