export const metadata = {
  title: "Privacy Policy VeinSight",
};

export default function PrivacyPolicy() {
  return (
    <main className="mx-auto min-h-dvh max-w-2xl bg-black px-6 py-12 text-neutral-300">
      <h1 className="text-2xl font-semibold text-white">Privacy Policy</h1>
      <p className="mt-2 text-sm text-neutral-500">Last updated: August 9, 2026</p>

      <Section title="What VeinSight does">
        <p>
          VeinSight uses your device&rsquo;s camera (or an uploaded image) to
          capture near infrared images of your forearm and processes them to
          highlight probable vein locations. It is a visualization aid, not a
          medical device. See our{" "}
          <a href="/terms" className="text-emerald-400 underline">
            Terms &amp; Disclaimer
          </a>
          .
        </p>
      </Section>

      <Section title="What we collect">
        <p>
          When you capture or upload an image, it is sent to our processing
          server to generate the vein overlay. We do not require an account,
          and we do not collect your name, contact details, or any
          identifying information as part of using the app.
        </p>
      </Section>

      <Section title="What happens to your images">
        <p>
          Images you capture or upload are processed in memory by our server
          to produce the result (arm isolation, vein overlay, and related
          analysis) and are not stored on our servers afterward.
        </p>
        <p className="mt-3">
          Anything saved after that point happens only on your own device,
          and only if you choose to:
        </p>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>
            <strong className="text-neutral-200">Save images</strong> writes
            the result images to your device&rsquo;s local storage (or
            triggers a download in your browser). This never touches our
            servers.
          </li>
          <li>
            <strong className="text-neutral-200">Save to history</strong>{" "}
            stores the result on your device only (using your browser or
            app&rsquo;s local storage). It is never uploaded anywhere, is not
            synced across devices, and is only visible on the device that
            saved it. You can delete individual saved items at any time from
            the History screen.
          </li>
        </ul>
      </Section>

      <Section title="Camera permission">
        <p>
          The app requests camera access solely to let you capture an image
          for processing. We do not record video, and no image leaves your
          device unless you actively capture or upload one for processing.
        </p>
      </Section>

      <Section title="Third parties">
        <p>
          We do not sell or share your images or usage data with third
          parties. We do not use advertising or analytics tracking in the
          app. Standard technical logs (e.g. server error logs, hosting
          provider access logs) may exist transiently as part of normal
          infrastructure operation, but are not used to identify you.
        </p>
      </Section>

      <Section title="Data deletion">
        <p>
          Since we do not retain your images on our servers, there is nothing
          to delete on our end after processing completes. To remove
          anything saved locally, delete it from the History screen in the
          app, or uninstall the app / clear site data in your browser.
        </p>
      </Section>

      <Section title="Children">
        <p>
          VeinSight is not directed at children and is not intended for use
          by anyone under 18.
        </p>
      </Section>

      <Section title="Changes to this policy">
        <p>
          If this policy changes, we will update the date at the top of this
          page.
        </p>
      </Section>

      <Section title="Contact">
        <p>
          Questions about this policy or your data: veinsight2026@gmail.com
        </p>
      </Section>
    </main>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-8">
      <h2 className="text-sm font-semibold uppercase tracking-[0.1em] text-neutral-400">
        {title}
      </h2>
      <div className="mt-2 text-sm leading-6">{children}</div>
    </section>
  );
}
