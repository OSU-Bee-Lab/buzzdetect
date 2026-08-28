// Test config kept separate from vite.config.js, which is tailored to Tauri's
// dev server and has no business being loaded by a test run.
//
// The svelte plugin is here because the stores under test are `.svelte.ts`
// files: runes ($state, and the getters that depend on them) are compiler
// syntax, so those files have to go through the Svelte compiler before node
// can run them. Tests that touch them are named `*.svelte.test.ts` for the
// same reason.
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [svelte({ hot: false })],
	resolve: {
		// The browser build of svelte is the one with a working reactivity
		// runtime outside a server render.
		conditions: ['browser']
	},
	test: {
		environment: 'node',
		include: ['src/**/*.{test,spec}.{js,ts}']
	}
});
