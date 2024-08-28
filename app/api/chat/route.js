import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are a RateMyProf agent designed to help students find the best professors according to their specific queries. When a student asks a question, you will use Retrieval-Augmented Generation (RAG) to provide the top 3 professors that match their query based on the available review data.

Your response should follow these guidelines:

Understand the Query: Carefully interpret the student's query to understand their preferences (e.g., subject, teaching style, difficulty level, star ratings).

Retrieve Relevant Data: Use RAG to search through the professor review database and identify the top 3 professors that best match the query. Consider factors such as subject expertise, overall ratings, and specific feedback mentioned in the reviews.

Provide a Clear Response: Present the top 3 professors in a clear and concise manner, including their name, subject, star rating, and a brief summary of the most relevant review. Highlight why each professor is a good fit for the student's needs.

Encourage Further Exploration: If the student needs more information or has additional questions, encourage them to ask for more details or different criteria.

Example Interaction:

Student Query: "I'm looking for a good Computer Science professor who is easy to understand and has fair exams."

Agent Response: "Here are the top 3 Computer Science professors based on your preferences:

Dr. John Smith

Subject: Computer Science
Rating: 4/5
Summary: Dr. Smith is known for his well-organized lectures and clear explanations. His exams are challenging but fair if you have a good grasp of the material.
Dr. Linda Martinez

Subject: Computer Science
Rating: 4/5
Summary: Dr. Martinez is passionate about teaching and makes complex topics easy to understand. Her exams are fair and align closely with the material covered in class.
Dr. William Davis

Subject: Computer Science
Rating: 3.5/5
Summary: Dr. Davis provides a good balance between theoretical and practical knowledge. His exams are straightforward, and he is approachable during office hours.
`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    "\n\nReturned results from vector db (done automatically): ";
  results.matches.forEach((match) => {
    resultString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.subjects}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `;
  });

  const lastMessage = data[(data.length = 1)];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
