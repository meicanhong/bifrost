package openrouter_test

import (
	"context"
	"encoding/json"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/providers/openrouter"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/internal/llmtests"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestOpenRouter(t *testing.T) {
	t.Parallel()
	if strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY")) == "" {
		t.Skip("Skipping OpenRouter tests because OPENROUTER_API_KEY is not set")
	}

	client, ctx, cancel, err := llmtests.SetupTest()
	if err != nil {
		t.Fatalf("Error initializing test setup: %v", err)
	}
	defer cancel()
	defer client.Shutdown()

	testConfig := llmtests.ComprehensiveTestConfig{
		Provider:             schemas.OpenRouter,
		ChatModel:            "openai/gpt-4.1",
		VisionModel:          "openai/gpt-4o",
		TextModel:            "google/gemini-2.5-flash",
		EmbeddingModel:       "qwen/qwen3-embedding-4b",
		ReasoningModel:       "openai/gpt-oss-120b",
		PromptCachingModel:   "anthropic/claude-sonnet-4", // Claude is the only OpenRouter model with explicit caching; its Responses half was broken until #6290
		TranscriptionModel:   "openai/gpt-4o-mini-transcribe",
		SpeechSynthesisModel: "minimax/speech-2.8-turbo",
		Scenarios: llmtests.TestScenarios{
			TextCompletion:             true,
			SimpleChat:                 true,
			CompletionStream:           true,
			MultiTurnConversation:      true,
			ToolCalls:                  true,
			ToolCallsStreaming:         false, // OpenRouter's responses API is in Beta
			MultipleToolCalls:          true,
			MultipleToolCallsStreaming: true,
			End2EndToolCalling:         true,
			AutomaticFunctionCall:      true,
			ImageURL:                   false, // OpenRouter's responses API is in Beta
			ImageBase64:                false, // OpenRouter's responses API is in Beta
			MultipleImages:             false, // OpenRouter's responses API is in Beta
			FileBase64:                 false, // Responses API times out (300s+) with file input
			FileURL:                    false, // Responses API times out (300s+) with file input
			CompleteEnd2End:            false, // OpenRouter's responses API is in Beta
			Reasoning:                  true,
			PromptCaching:              true, // Gates the three tool-block scenarios; two of them issue Responses requests, which is what #6290 broke
			ListModels:                 true,
			StructuredOutputs:          true, // Structured outputs with nullable enum support
			Embedding:                  true,
			SpeechSynthesis:            true,  // Supported via OpenAI-compatible /v1/audio/speech
			SpeechSynthesisStream:      false, // Streaming not offered by upstream OpenRouter API
			Transcription:              true,  // Supported via OpenAI-compatible /v1/audio/transcriptions
			TranscriptionStream:        false, // Streaming not offered by upstream OpenRouter API
		},
	}

	t.Run("OpenRouterTests", func(t *testing.T) {
		llmtests.RunAllComprehensiveTests(t, client, ctx, testConfig)
	})
}

// TestOpenRouterRerank exercises the native provider against the upstream wire contract.
func TestOpenRouterRerank(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" || r.Method != "POST" || r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "voyageai/rerank-3-lite" || body["query"] != "fruit" || body["top_n"] != float64(1) {
			t.Errorf("unexpected body: %v", body)
		}
		docs, ok := body["documents"].([]interface{})
		if !ok || len(docs) != 2 {
			t.Errorf("unexpected documents: %v", body["documents"])
		} else if doc, ok := docs[1].(map[string]interface{}); !ok || doc["text"] != "apple" {
			t.Errorf("unexpected document: %v", docs[1])
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"rerank-test","results":[{"index":1,"relevance_score":0.9}],"usage":{"total_tokens":150,"search_units":1,"cost":0.00001}}`))
	}))
	defer server.Close()
	provider := openrouter.NewOpenRouterProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, AllowPrivateNetwork: true}}, bifrost.NewDefaultLogger(schemas.LogLevelError))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	result, err := provider.Rerank(ctx, schemas.Key{Value: schemas.SecretVar{Val: "test-key"}}, &schemas.BifrostRerankRequest{Model: "voyageai/rerank-3-lite", Query: "fruit", Documents: []schemas.RerankDocument{{Text: "car"}, {Text: "apple"}}, Params: &schemas.RerankParameters{TopN: schemas.Ptr(1), ReturnDocuments: schemas.Ptr(true)}})
	if err != nil {
		t.Fatalf("rerank failed: %+v", err)
	}
	if len(result.Results) != 1 || result.Results[0].Index != 1 || result.Results[0].Document == nil || result.Results[0].Document.Text != "apple" {
		t.Fatalf("unexpected results: %+v", result)
	}
	if result.Usage == nil || result.Usage.TotalTokens != 150 || result.Usage.SearchUnits == nil || *result.Usage.SearchUnits != 1 || result.Usage.Cost == nil || result.Usage.Cost.TotalCost != 0.00001 {
		t.Fatalf("lost usage: %+v", result.Usage)
	}
}

func TestOpenRouterRerankModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("output_modalities") == "rerank" {
			w.Write([]byte(`{"data":[{"id":"voyageai/rerank-3-lite"},{"id":"voyageai/rerank-3-lite"}]}`))
			return
		}
		w.Write([]byte(`{"data":[]}`))
	}))
	defer server.Close()
	provider := openrouter.NewOpenRouterProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, AllowPrivateNetwork: true}}, bifrost.NewDefaultLogger(schemas.LogLevelError))
	for _, blocked := range []bool{false, true} {
		key := schemas.Key{Value: schemas.SecretVar{Val: "test-key"}, Models: schemas.WhiteList{"*"}}
		if blocked {
			key.BlacklistedModels = schemas.BlackList{"voyageai/rerank-3-lite"}
		}
		result, err := provider.ListModels(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), []schemas.Key{key}, &schemas.BifrostListModelsRequest{})
		if err != nil {
			t.Fatalf("list failed: %+v", err)
		}
		want := 1
		if blocked {
			want = 0
		}
		if len(result.Data) != want {
			t.Fatalf("blocked=%v: expected %d models, got %+v", blocked, want, result.Data)
		}
		if !blocked && result.Data[0].ID != "openrouter/voyageai/rerank-3-lite" {
			t.Fatalf("wrong ID: %s", result.Data[0].ID)
		}
	}
}

func TestOpenRouterRerankUpstreamError(t *testing.T) {
	for _, status := range []int{401, 429, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(status)
				w.Write([]byte(`{"error":{"message":"upstream rejected request","type":"provider_error"}}`))
			}))
			defer server.Close()
			provider := openrouter.NewOpenRouterProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, AllowPrivateNetwork: true}}, bifrost.NewDefaultLogger(schemas.LogLevelError))
			result, err := provider.Rerank(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), schemas.Key{}, &schemas.BifrostRerankRequest{Model: "voyageai/rerank-3-lite", Query: "fruit", Documents: []schemas.RerankDocument{{Text: "apple"}}})
			if result != nil || err == nil || err.StatusCode == nil || *err.StatusCode != status {
				t.Fatalf("expected HTTP %d, got response=%+v error=%+v", status, result, err)
			}
		})
	}
}

func TestOpenRouterSupplementalCatalogFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("output_modalities") == "rerank" {
			w.WriteHeader(503)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/embeddings/models" {
			w.Write([]byte(`{"data":[{"id":"qwen/embedding"}]}`))
			return
		}
		w.Write([]byte(`{"data":[{"id":"openai/chat"}]}`))
	}))
	defer server.Close()
	provider := openrouter.NewOpenRouterProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, AllowPrivateNetwork: true}}, bifrost.NewDefaultLogger(schemas.LogLevelError))
	result, err := provider.ListModels(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), []schemas.Key{{Models: schemas.WhiteList{"*"}}}, &schemas.BifrostListModelsRequest{})
	if err != nil || len(result.Data) != 2 {
		t.Fatalf("primary and embedding catalogs must survive: result=%+v err=%+v", result, err)
	}
}

func TestOpenRouterRerankMalformedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{not-json`)) }))
	defer server.Close()
	provider := openrouter.NewOpenRouterProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, AllowPrivateNetwork: true}}, bifrost.NewDefaultLogger(schemas.LogLevelError))
	result, err := provider.Rerank(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), schemas.Key{}, &schemas.BifrostRerankRequest{Model: "voyageai/rerank-3-lite", Query: "fruit", Documents: []schemas.RerankDocument{{Text: "apple"}}})
	if result != nil || err == nil {
		t.Fatalf("malformed upstream response must fail: result=%+v err=%+v", result, err)
	}
}
