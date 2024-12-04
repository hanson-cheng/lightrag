-- Enable the vector extension
create extension if not exists vector;

-- Create documents table
create table if not exists documents (
    id text primary key,
    content text not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    embedding vector(1536)  -- For future OpenAI embeddings support
);

-- Create a function to update the embedding
create or replace function update_embedding()
returns trigger
language plpgsql
as $$
begin
    -- In the future, we can add embedding generation logic here
    return new;
end;
$$;

-- Create a trigger to automatically update embeddings
create trigger update_document_embedding
    before insert or update on documents
    for each row
    execute function update_embedding();
