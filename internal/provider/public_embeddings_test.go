package provider

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/yetone/magpie/internal/catalog"
)

// OpenRouter publishes its embedding models on a list of their own,
// /api/v1/embeddings/models, and not in the /models a key is asked at. The
// list kept for a provider there has them, marked as serving embeddings, so
// /v1/models?kind=embedding names them and an agent's embedding route can
// pick one (#7321).
func TestPublicEmbeddingsJoinTheList(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	emb, err := os.ReadFile(filepath.Join("..", "catalog", "testdata", "openrouter-embeddings-2026-11-20.json"))
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/v1/embeddings/models" {
			if r.Header.Get("Authorization") != "" {
				t.Errorf("the public list was sent the key")
			}
			w.Write(emb)
			return
		}
		w.Write([]byte(`{"data":[{"id":"openai/gpt-6"}]}`))
	}))
	defer server.Close()
	old := catalog.PublicEmbeddingCatalogs
	catalog.PublicEmbeddingCatalogs = map[string]string{"127.0.0.1": server.URL + "/api/v1/embeddings/models"}
	t.Cleanup(func() { catalog.PublicEmbeddingCatalogs = old })

	p := Provider{ID: "or", Chat: server.URL + "/v1", Key: "k"}
	chat, err := p.Fetch(context.Background())
	if err != nil || len(chat) != 1 || chat[0].ID != "openai/gpt-6" {
		t.Fatalf("chat models %+v, %v", chat, err)
	}
	served := map[string]catalog.Model{}
	var ids []string
	for _, m := range catalog.LiveServing("or", catalog.ServesEmbedding) {
		served[m.ID] = m
		ids = append(ids, m.ID)
	}
	if len(ids) != 38 {
		t.Fatalf("%d embedding models kept: %v", len(ids), ids)
	}
	for _, id := range ids {
		if served[id].Serves != catalog.ServesEmbedding || served[id].Draws || served[id].Films {
			t.Errorf("%s kept as %+v", id, served[id])
		}
	}
	// the vendor's prefix is the provider's to say, and a multimodal one
	// takes images as its list says
	if m := served["cohere/embed-5-fast"]; m.Name != "Embed 5 Fast" || m.Context != 128000 || !m.Images || m.Released != "2026-11-03" {
		t.Errorf("cohere/embed-5-fast kept as %+v", m)
	}
	if m := served["amazon/titan-embed-text-v1"]; m.Images || m.Context != 8192 {
		t.Errorf("amazon/titan-embed-text-v1 kept as %+v", m)
	}
	if _, ok := served["nvidia/nemotron-3.5-embed-1b:free"]; !ok {
		t.Error("a free embedding model was left out")
	}
}
