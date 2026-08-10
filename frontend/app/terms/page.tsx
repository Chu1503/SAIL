export const metadata = {
  title: "Terms & Disclaimer VeinSight",
};

export default function Terms() {
  return (
    <main className="selectable-text mx-auto min-h-dvh max-w-2xl bg-black px-6 py-12 text-neutral-300">
      <h1 className="text-2xl font-semibold text-white">
        Terms of Use &amp; Medical Disclaimer
      </h1>
      <p className="mt-2 text-sm text-neutral-500">Last updated: August 9, 2026</p>

      <Section title="Not a medical device">
        <p>
          VeinSight is a visualization aid that highlights probable vein
          locations from a near infrared image using automated image
          processing and machine learning. It is{" "}
          <strong className="text-neutral-200">
            not a certified or regulated medical device
          </strong>
          , has not been evaluated by any medical regulatory body (e.g. FDA,
          CE), and is not intended to diagnose, treat, or guide any medical
          procedure on its own.
        </p>
      </Section>

      <Section title="No substitute for training or judgment">
        <p>
          Any point, mark, or overlay VeinSight shows, including the
          suggested injection point, is an algorithmic estimate based on
          image analysis. It can be wrong, incomplete, or misleading,
          especially on images that differ from what the underlying model
          was trained on. It is not a substitute for the judgment,
          training, or assessment of a qualified healthcare professional.
          Do not use VeinSight as the sole basis for performing venipuncture
          or any medical procedure on yourself or anyone else.
        </p>
      </Section>

      <Section title="Your responsibility">
        <p>
          You are solely responsible for how you use this app and any
          actions you take based on its output. Use of VeinSight for actual
          medical procedures should only be done by, or under the
          supervision of, someone qualified to perform that procedure.
        </p>
      </Section>

      <Section title="No warranty">
        <p>
          VeinSight is provided &ldquo;as is,&rdquo; without warranties of
          any kind, express or implied, including accuracy, reliability, or
          fitness for a particular purpose.
        </p>
      </Section>

      <Section title="Limitation of liability">
        <p>
          To the fullest extent permitted by law, we are not liable for any
          injury, harm, or damages arising from use or misuse of this app or
          reliance on its output.
        </p>
      </Section>

      <Section title="Age requirement">
        <p>You must be 18 or older to use VeinSight.</p>
      </Section>

      <Section title="Changes">
        <p>
          We may update these terms from time to time. Continued use of the
          app after changes means you accept the updated terms.
        </p>
      </Section>

      <Section title="Contact">
        <p>Questions about these terms: veinsight2026@gmail.com</p>
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
